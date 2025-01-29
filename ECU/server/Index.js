/**
 * @file server.js
 * @description This file initializes and starts the Node.js server, serves the frontend, and integrates the WebSocket server.
 * @author Lior Jigalo
 * @license MIT
 */

/** Import application module */
const app = require("./app");

/** Load environment variables from a .env file */
require("dotenv").config();

/**
 * @typedef {Object} ProcessEnv
 * @property {string} PORT - Port number for the server to listen on.
 * @property {string} IP - IP address for the server to bind to.
 */

/** @type {ProcessEnv}*/
const {PORT, IP} = process.env;

/** Import utility functions for drive log management */
const {} = require("./util/driveLogManager");

/** Import WebSocket initialization function */
const { initWebSocket, connectVideoFeedWebSocket} = require("./websocket");

/** Import Express framework */
const express = require("express");

/** Import path module for handling file and directory paths */
const path = require("path");
const {readdir} = require("node:fs");

/**
 * @function startApp
 * @description Initializes the server, serves static frontend files, and integrates the WebSocket server.
 */
const startApp = () => {
    /**
     * Serve static files from the 'frontend' directory.
     */
    app.use(express.static(path.join(__dirname, "frontend")));

    /**
     * @function app.get
     * @description Serve the main HTML file for the root route.
     * @param {string} path - URL path for the route.
     * @param {function} callback - Callback function handling HTTP requests and responses.
     */
    app.get("/", (req, res) => {
        res.sendFile(path.join(__dirname, "frontend", "index.html"));
    });

    // Statically serve your "Saved_videos" folder at the /Saved_videos URL
    app.use("/Saved_videos", express.static(path.join(__dirname, "Saved_videos")));

    // Endpoint to get the video file list
    app.get("/api/videos", (req, res) => {
        const videosDir = path.join(__dirname, "Saved_videos");
        readdir(videosDir, (err, files) => {
            if (err) {
                return res.status(500).json({ error: "Error reading directory" });
            }
            // Filter for common video extensions only if you want
            const videoFiles = files.filter(file =>
                file.match(/\.(mp4|mov|avi|webm|mkv)$/i)
            );
            res.json(videoFiles);
        });
    });

    /**
     * @function app.listen
     * @description Creates an HTTP server and listens on the specified port and IP address.
     * @param {number} PORT - Port number for the server.
     * @param {string} IP - IP address for the server to bind to.
     * @param {function} callback - Callback function executed when the server starts.
     */
    const server = app.listen(PORT, `${IP}`, () => {
        console.log(`Server running on port ${PORT} and listening on 0.0.0.0`);
    });

    /**
     * Initialize WebSocket server on the same HTTP server.
     * @param {object} server - The HTTP server instance to attach WebSocket.
     */
    initWebSocket(server);

    /**
     * Initialize WebSocket server that receives video feed.
     */
    connectVideoFeedWebSocket();
};

// Start the application
startApp();