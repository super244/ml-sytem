const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("aiFactoryDesktop", {
  platform: process.platform,
  desktopShell: "electron",
  version: process.env.npm_package_version || "1.0.0",
  
  // System integration
  showNotification: (title, body) => {
    return new Promise((resolve) => {
      ipcRenderer.invoke('show-notification', { title, body })
        .then(resolve)
        .catch(() => resolve(false));
    });
  },
  
  // File operations
  selectFile: (filters) => {
    return ipcRenderer.invoke('select-file', filters);
  },
  
  selectDirectory: () => {
    return ipcRenderer.invoke('select-directory');
  },
  
  // App info
  getAppVersion: () => {
    return ipcRenderer.invoke('get-app-version');
  },
  
  // Window controls
  minimizeWindow: () => {
    ipcRenderer.send('minimize-window');
  },
  
  maximizeWindow: () => {
    ipcRenderer.send('maximize-window');
  },
  
  closeWindow: () => {
    ipcRenderer.send('close-window');
  }
});

// Error handling for unhandled promises
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
});
