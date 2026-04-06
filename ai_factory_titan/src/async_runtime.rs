//! Async runtime with work stealing for the Titan engine.
//!
//! Provides a high-performance task executor with:
//! - Work-stealing scheduler for load balancing
//! - Task prioritization (High/Normal/Background)
//! - Async/await support for kernel execution
//! - Cancellation support

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake};

use crossbeam_queue::SegQueue;
use parking_lot::Mutex;
use tokio::sync::oneshot;

/// Priority levels for task scheduling
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    High = 2,
    Normal = 1,
    Background = 0,
}

/// Task handle for awaiting completion and cancellation
pub struct TaskHandle<T> {
    rx: oneshot::Receiver<T>,
    id: TaskId,
    scheduler: Arc<WorkStealingScheduler>,
}

impl<T> TaskHandle<T> {
    /// Get the task ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Cancel the task
    pub fn cancel(&self) {
        self.scheduler.cancel_task(self.id);
    }
}

impl<T> Future for TaskHandle<T> {
    type Output = Option<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.rx.try_recv() {
            Ok(value) => Poll::Ready(Some(value)),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => Poll::Ready(None),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

/// Unique task identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// A unit of work in the async runtime
pub struct Task {
    id: TaskId,
    priority: TaskPriority,
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl Task {
    /// Create a new task from a future
    pub fn new<F>(future: F, priority: TaskPriority) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        Self {
            id: TaskId::next(),
            priority,
            future: Mutex::new(Box::pin(future)),
        }
    }

    /// Poll the task future
    fn poll(&self, context: &mut Context) -> Poll<()> {
        self.future.lock().as_mut().poll(context)
    }
}

/// Work-stealing task scheduler
pub struct WorkStealingScheduler {
    high_priority: SegQueue<Arc<Task>>,
    normal_priority: SegQueue<Arc<Task>>,
    background_priority: SegQueue<Arc<Task>>,
    cancelled: Mutex<std::collections::HashSet<TaskId>>,
    task_count: AtomicU64,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Submit a task to the scheduler
    pub fn submit<F, T>(self: &Arc<Self>, future: F, priority: TaskPriority) -> TaskHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let id = TaskId::next();

        let wrapped_future = async move {
            let result = future.await;
            let _ = tx.send(result);
        };

        let task = Arc::new(Task {
            id,
            priority,
            future: Mutex::new(Box::pin(wrapped_future)),
        });

        self.task_count.fetch_add(1, Ordering::SeqCst);

        match priority {
            TaskPriority::High => self.high_priority.push(task),
            TaskPriority::Normal => self.normal_priority.push(task),
            TaskPriority::Background => self.background_priority.push(task),
        }

        TaskHandle {
            rx,
            id,
            scheduler: Arc::clone(self),
        }
    }

    /// Steal a task from the scheduler
    pub fn steal_task(&self) -> Option<Arc<Task>> {
        // Priority order: High > Normal > Background
        self.high_priority
            .pop()
            .or_else(|| self.normal_priority.pop())
            .or_else(|| self.background_priority.pop())
    }

    /// Cancel a task by ID
    pub fn cancel_task(&self, id: TaskId) {
        self.cancelled.lock().insert(id);
    }

    /// Check if a task is cancelled
    pub fn is_cancelled(&self, id: TaskId) -> bool {
        self.cancelled.lock().contains(&id)
    }

    /// Get the current task count
    pub fn task_count(&self) -> u64 {
        self.task_count.load(Ordering::SeqCst)
    }
}

impl Default for WorkStealingScheduler {
    fn default() -> Self {
        Self {
            high_priority: SegQueue::new(),
            normal_priority: SegQueue::new(),
            background_priority: SegQueue::new(),
            cancelled: Mutex::new(std::collections::HashSet::new()),
            task_count: AtomicU64::new(0),
        }
    }
}

/// Thread-local task queue for work stealing
thread_local! {
    static LOCAL_QUEUE: std::cell::RefCell<Vec<Arc<Task>>> = std::cell::RefCell::new(Vec::new());
}

/// High-performance async executor for the Titan engine
pub struct TitanExecutor {
    scheduler: Arc<WorkStealingScheduler>,
    num_workers: usize,
}

impl TitanExecutor {
    /// Create a new executor with the specified number of worker threads
    pub fn new(num_workers: usize) -> Self {
        Self {
            scheduler: WorkStealingScheduler::new(),
            num_workers,
        }
    }

    /// Spawn a task onto the executor
    pub fn spawn<F, T>(&self, future: F) -> TaskHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.scheduler.submit(future, TaskPriority::Normal)
    }

    /// Spawn a high-priority task
    pub fn spawn_high<F, T>(&self, future: F) -> TaskHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.scheduler.submit(future, TaskPriority::High)
    }

    /// Spawn a background task
    pub fn spawn_background<F, T>(&self, future: F) -> TaskHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.scheduler.submit(future, TaskPriority::Background)
    }

    /// Run the executor (blocking)
    pub fn run(&self) {
        let mut handles = vec![];

        for _ in 0..self.num_workers {
            let scheduler = Arc::clone(&self.scheduler);
            let handle = std::thread::spawn(move || {
                Worker::new(scheduler).run();
            });
            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }
    }

    /// Get the scheduler reference
    pub fn scheduler(&self) -> &Arc<WorkStealingScheduler> {
        &self.scheduler
    }
}

/// Worker thread that steals and executes tasks
struct Worker {
    scheduler: Arc<WorkStealingScheduler>,
}

impl Worker {
    fn new(scheduler: Arc<WorkStealingScheduler>) -> Self {
        Self { scheduler }
    }

    fn run(&self) {
        loop {
            if let Some(task) = self.scheduler.steal_task() {
                if !self.scheduler.is_cancelled(task.id) {
                    self.execute_task(&task);
                }
            } else {
                // No tasks available, yield to avoid busy-waiting
                std::thread::yield_now();
            }
        }
    }

    fn execute_task(&self, task: &Arc<Task>) {
        struct TaskWaker {
            task: Arc<Task>,
            scheduler: Arc<WorkStealingScheduler>,
        }

        impl Wake for TaskWaker {
            fn wake(self: Arc<Self>) {
                // Re-schedule the task
                match self.task.priority {
                    TaskPriority::High => self.scheduler.high_priority.push(Arc::clone(&self.task)),
                    TaskPriority::Normal => self.scheduler.normal_priority.push(Arc::clone(&self.task)),
                    TaskPriority::Background => self.scheduler.background_priority.push(Arc::clone(&self.task)),
                }
            }
        }

        let waker = Arc::new(TaskWaker {
            task: Arc::clone(task),
            scheduler: Arc::clone(&self.scheduler),
        })
        .into();
        let mut context = Context::from_waker(&waker);

        let _ = task.poll(&mut context);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_task_id_generation() {
        let id1 = TaskId::next();
        let id2 = TaskId::next();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_scheduler_submit() {
        let scheduler = WorkStealingScheduler::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let handle = scheduler.submit(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            42
        }, TaskPriority::Normal);

        assert_eq!(handle.id().0, 1); // First task should have ID 1
    }
}
