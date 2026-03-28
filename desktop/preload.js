const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("aiFactoryDesktop", {
  platform: process.platform,
  desktopShell: "electron",
});
