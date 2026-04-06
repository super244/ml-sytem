const { app, BrowserWindow, Menu, Notification, dialog, ipcMain, shell } = require("electron");
const path = require("path");

const DEFAULT_URL = process.env.AI_FACTORY_DESKTOP_URL ?? "http://127.0.0.1:3000/dashboard";

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

  window.once("ready-to-show", () => {
    window.show();
  });

  const loadWithRetry = () => {
    window.loadURL(DEFAULT_URL).catch((err) => {
      console.log("Failed to load dashboard, retrying in 2s...", err.message);
      setTimeout(loadWithRetry, 2000);
    });
  };
  loadWithRetry();

  if (process.env.NODE_ENV === "development") {
    window.webContents.openDevTools({ mode: "detach" });
  }

  window.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  window.webContents.on("did-fail-load", (event, errorCode, errorDescription) => {
    if (errorCode === -102 || errorCode === -7) {
      console.log(`Connection refused (${errorDescription}), auto-retrying...`);
      setTimeout(loadWithRetry, 2000);
    }
  });
}

function createMenu() {
  const template = [
    {
      label: "AI-Factory",
      submenu: [
        {
          label: "About AI-Factory",
          click: () => {
            shell.openExternal("https://github.com/super244/ai-factory");
          },
        },
        { type: "separator" },
        {
          label: "Quit",
          accelerator: process.platform === "darwin" ? "Cmd+Q" : "Ctrl+Q",
          click: () => {
            app.quit();
          },
        },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "close" },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

function getFocusedWindow() {
  return BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows()[0] ?? null;
}

ipcMain.handle("show-notification", (_event, payload) => {
  const notification = new Notification({
    title: payload?.title ?? "AI-Factory",
    body: payload?.body ?? "",
  });
  notification.show();
  return true;
});

ipcMain.handle("select-file", async (_event, filters) => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters,
  });
  return result.canceled ? null : result.filePaths[0] ?? null;
});

ipcMain.handle("select-directory", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
  });
  return result.canceled ? null : result.filePaths[0] ?? null;
});

ipcMain.handle("get-app-version", () => app.getVersion());

ipcMain.on("minimize-window", () => {
  getFocusedWindow()?.minimize();
});

ipcMain.on("maximize-window", () => {
  const window = getFocusedWindow();
  if (!window) {
    return;
  }
  if (window.isMaximized()) {
    window.unmaximize();
  } else {
    window.maximize();
  }
});

ipcMain.on("close-window", () => {
  getFocusedWindow()?.close();
});

app.whenReady().then(() => {
  createWindow();
  createMenu();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
      app.quit();
    }
  });

  app.setAsDefaultProtocolClient("ai-factory");
});

app.on("web-contents-created", (_event, contents) => {
  contents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
});
