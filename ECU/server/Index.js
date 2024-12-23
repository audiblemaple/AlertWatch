const app = require("./app");
require("dotenv").config();
const {PORT, IP} = process.env;
const {} = require("./util/driveLogManager");
const { initWebSocket } = require("./websocket");
const express = require("express");
const path = require("path");

const startApp = () => {

    app.use(express.static(path.join(__dirname, "frontend")));

    app.get("/", (req, res) => {
        res.sendFile(path.join(__dirname, "frontend", "index.html"));
    });

    // Create HTTP server and listen on 0.0.0.0 and specified PORT
    const server = app.listen(PORT, `${IP}`, () => {
        console.log(`Server running on port ${PORT} and listening on 0.0.0.0`);
    });

    // Initialize WebSocket server on the same HTTP server
    initWebSocket(server);
};

startApp();