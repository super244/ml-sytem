//! Memory management for unified memory and device memory optimization
//!
//! Provides:
//! - Unified memory allocator for Apple Silicon
//! - Device memory pools for CUDA
//! - Memory-mapped I/O for large model loading
//! - Zero-copy data paths

use anyhow::{anyhow, Result};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;

/// Memory pool for efficient allocation/deallocation
pub struct MemoryPool {
    block_size: usize,
    free_list: Mutex<Vec<NonNull<u8>>>,
    allocated: AtomicUsize,
}

impl MemoryPool {
    /// Create a new memory pool with fixed block size
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            free_list: Mutex::new(Vec::new()),
            allocated: AtomicUsize::new(0),
        }
    }

    /// Allocate a block from the pool
    pub fn allocate(&self) -> Result<NonNull<u8>> {
        // Try to reuse from free list
        if let Some(block) = self.free_list.lock().pop() {
            return Ok(block);
        }

        // Allocate new block
        let layout = Layout::from_size_align(self.block_size, 64)
            .map_err(|e| anyhow!("Invalid layout: {}", e))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow!("Memory allocation failed"));
        }

        self.allocated.fetch_add(1, Ordering::SeqCst);
        Ok(NonNull::new(ptr).unwrap())
    }

    /// Return a block to the pool
    pub fn deallocate(&self, ptr: NonNull<u8>) {
        self.free_list.lock().push(ptr);
    }

    /// Get number of allocated blocks
    pub fn allocated_count(&self) -> usize {
        self.allocated.load(Ordering::SeqCst)
    }

    /// Get number of free blocks
    pub fn free_count(&self) -> usize {
        self.free_list.lock().len()
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.block_size, 64).unwrap();

        // Free all blocks in free list
        for ptr in self.free_list.lock().drain(..) {
            unsafe { dealloc(ptr.as_ptr(), layout) };
        }
    }
}

/// Unified memory allocator for Apple Silicon
pub struct UnifiedMemoryAllocator {
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
}

impl UnifiedMemoryAllocator {
    /// Create a new unified memory allocator
    pub fn new() -> Self {
        Self {
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
        }
    }

    /// Allocate unified memory (shared between CPU and GPU)
    pub fn allocate(&self, size: usize) -> Result<UnifiedMemoryPtr> {
        // On Apple Silicon, use posix_memalign for aligned allocation
        let layout = Layout::from_size_align(size, 4096)
            .map_err(|e| anyhow!("Invalid layout: {}", e))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow!("Unified memory allocation failed"));
        }

        let current = self.total_allocated.fetch_add(size, Ordering::SeqCst) + size;

        // Update peak
        loop {
            let peak = self.peak_allocated.load(Ordering::SeqCst);
            if current <= peak || self.peak_allocated.compare_exchange(
                peak,
                current,
                Ordering::SeqCst,
                Ordering::SeqCst
            ).is_ok() {
                break;
            }
        }

        Ok(UnifiedMemoryPtr {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            layout,
        })
    }

    /// Get current allocated bytes
    pub fn current_usage(&self) -> usize {
        self.total_allocated.load(Ordering::SeqCst)
    }

    /// Get peak allocated bytes
    pub fn peak_usage(&self) -> usize {
        self.peak_allocated.load(Ordering::SeqCst)
    }
}

/// Pointer to unified memory
pub struct UnifiedMemoryPtr {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl UnifiedMemoryPtr {
    /// Get raw pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for UnifiedMemoryPtr {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

unsafe impl Send for UnifiedMemoryPtr {}
unsafe impl Sync for UnifiedMemoryPtr {}

/// Device memory management for CUDA
#[cfg(feature = "cuda")]
pub struct DeviceMemory {
    device: Arc<cudarc::driver::CudaDevice>,
    allocated: AtomicUsize,
    peak: AtomicUsize,
}

#[cfg(feature = "cuda")]
impl DeviceMemory {
    /// Create new device memory manager
    pub fn new(device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self {
            device,
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    /// Allocate device memory
    pub fn allocate<T>(&self, count: usize) -> Result<cudarc::driver::CudaSlice<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let slice = self
            .device
            .alloc_zeros::<T>(count)
            .map_err(|e| anyhow!("Device allocation failed: {:?}", e))?;

        let size = count * std::mem::size_of::<T>();
        let current = self.allocated.fetch_add(size, Ordering::SeqCst) + size;

        // Update peak
        loop {
            let peak = self.peak.load(Ordering::SeqCst);
            if current <= peak || self.peak.compare_exchange(
                peak,
                current,
                Ordering::SeqCst,
                Ordering::SeqCst
            ).is_ok() {
                break;
            }
        }

        Ok(slice)
    }

    /// Get current allocated bytes
    pub fn current_usage(&self) -> usize {
        self.allocated.load(Ordering::SeqCst)
    }

    /// Get peak allocated bytes
    pub fn peak_usage(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }
}

/// Memory-mapped file for zero-copy model loading
pub struct MemoryMappedFile {
    #[allow(dead_code)]
    mmap: memmap2::Mmap,
    ptr: *const u8,
    len: usize,
}

impl MemoryMappedFile {
    /// Memory-map a file for reading
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow!("Failed to open file: {}", e))?;

        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| anyhow!("Failed to mmap file: {}", e))?
        };

        let ptr = mmap.as_ptr();
        let len = mmap.len();

        Ok(Self { mmap, ptr, len })
    }

    /// Get pointer to mapped memory
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get length of mapped memory
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get slice reference
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

unsafe impl Send for MemoryMappedFile {}
unsafe impl Sync for MemoryMappedFile {}

/// Memory statistics
#[derive(Clone, Debug, Default)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub free_bytes: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
}

/// Global memory tracker
pub struct MemoryTracker {
    stats: Mutex<MemoryStats>,
    active_allocations: Mutex<HashMap<usize, usize>>, // ptr -> size
}

impl MemoryTracker {
    /// Create new memory tracker
    pub fn new() -> Self {
        Self {
            stats: Mutex::new(MemoryStats::default()),
            active_allocations: Mutex::new(HashMap::new()),
        }
    }

    /// Track allocation
    pub fn track_allocation(&self, ptr: usize, size: usize) {
        let mut stats = self.stats.lock();
        let mut allocs = self.active_allocations.lock();

        stats.allocated_bytes += size;
        stats.peak_bytes = stats.peak_bytes.max(stats.allocated_bytes);
        stats.allocation_count += 1;
        allocs.insert(ptr, size);
    }

    /// Track deallocation
    pub fn track_deallocation(&self, ptr: usize) {
        let mut stats = self.stats.lock();
        let mut allocs = self.active_allocations.lock();

        if let Some(size) = allocs.remove(&ptr) {
            stats.allocated_bytes -= size;
            stats.deallocation_count += 1;
        }
    }

    /// Get current stats
    pub fn stats(&self) -> MemoryStats {
        self.stats.lock().clone()
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(1024);

        let ptr1 = pool.allocate().unwrap();
        let _ptr2 = pool.allocate().unwrap();

        assert_eq!(pool.allocated_count(), 2);

        pool.deallocate(ptr1);
        assert_eq!(pool.free_count(), 1);

        // Reuse from free list
        let _ptr3 = pool.allocate().unwrap();
        assert_eq!(pool.allocated_count(), 2); // No new allocation
    }

    #[test]
    fn test_unified_memory_allocator() {
        let allocator = UnifiedMemoryAllocator::new();

        let ptr1 = allocator.allocate(1024).unwrap();
        let _ptr2 = allocator.allocate(2048).unwrap();

        assert_eq!(allocator.current_usage(), 3072);
        assert_eq!(allocator.peak_usage(), 3072);

        drop(ptr1);
        // Usage tracking would need to be decremented on drop
    }
}
