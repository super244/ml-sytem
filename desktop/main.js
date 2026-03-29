const { app, BrowserWindow, Menu, ipcMain } = require("electron");
const path = require("path");

const DEFAULT_URL = process.env.AI_FACTORY_DESKTOP_URL ?? "http://127.0.0.1:3000/workspace";

function createWindow() {
  const window = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    autoHideMenuBar: true,
    backgroundColor: "#0f1317",
    title: "AI-Factory - Unified AI Platform",
    icon: path.join(__dirname, "assets", "icon.png"),
    show: false, // Don't show until ready-to-show
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      enableRemoteModule: false,
      webSecurity: true,
    },
  });

  // Show window when ready to prevent visual flash
  window.once('ready-to-show', () => {
    window.show();
  });

  window.loadURL(DEFAULT_URL);
  
  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    window.webContents.openDevTools();
  }
}

function createMenu() {
  const template = [
    {
      label: 'AI-Factory',
      submenu: [
        {
          label: 'About AI-Factory',
          click: () => {
            // Show about dialog
          }
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'close' }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

app.whenReady().then(() => {
  createWindow();
  createMenu();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

// Handle app protocol for deep linking
app.setAsDefaultProtocolClient('ai-factory');

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (navigationEvent, navigationURL) => {
    navigationEvent.preventDefault();
    require('electron').shell.openExternal(navigationURL);
  });
});
