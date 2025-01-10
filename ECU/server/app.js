/**
 * @file app.js
 * @description This file sets up the Express application, configures middleware, and establishes API routes.
 * @author Lior Jigalo
 * @license MIT
 */

/** Import Express framework */
const express = require("express");

/** Middleware for parsing JSON request bodies */
const bodyParser = express.json;

/** Middleware for enabling Cross-Origin Resource Sharing (CORS) */
const cors = require("cors");

/** Create an instance of the Express application */
const app = express();

/**
 * Enable CORS to allow cross-origin requests.
 */
app.use(cors());
/**
 * Parse incoming JSON request bodies.
 */
app.use(bodyParser());

/**
 * Export the configured Express application instance.
 */
module.exports = app;