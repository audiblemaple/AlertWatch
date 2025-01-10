/**
 * @file driveLogManager.js
 * @description Manages the creation and updating of drive log files with timestamped filenames.
 * @author Lior Jigalo
 * @license MIT
 */

/** Import file system and path modules */
const fs = require("fs");
const path = require("path");

/** Current drive log filename */
let currentDriveLogName = "";

/** Current drive object containing log data */
let currentDriveObject = {};


/**
 * Directory for storing log files (relative to this module's __dirname).
 * Ensures the logs directory exists.
 * @type {string}
 */
const logsDir = path.join(__dirname, "../logs");
if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
}

/**
 * Generates a custom timestamped filename for the log file.
 * @returns {string} Timestamped filename.
 */
const generateDriveFileName = () => {
    const date = new Date();
    const seconds= String(date.getSeconds()).padStart(2, "0");
    const minutes= String(date.getMinutes()).padStart(2, "0");
    const hours  = String(date.getHours()).padStart(2, "0");
    const day    = String(date.getDate()).padStart(2, "0");
    const month  = String(date.getMonth() + 1).padStart(2, "0"); // Month is zero-based
    const year  = date.getFullYear();

    // Return just the file name. We'll prepend logsDir when creating the filePath.
    return `Drive-${seconds}-${minutes}-${hours}T${day}-${month}-${year}.json`;
};

/**
 * Creates a log file with a timestamped filename and initializes it with default data.
 */
const createDriveLogFile = () => {
    const fileName = generateDriveFileName();
    // Use logsDir to build the absolute file path
    const filePath = path.join(logsDir, fileName);

    currentDriveLogName = fileName;

    const initialData = {
        timestamp: new Date().toISOString(),
        alert_count: 0,
        mild_alert_num: 0,
        medium_alert_num: 0,
        high_alert_num: 0,
        consecutive_alert_num: 0,
        consecutive_cant_parse: 0,
    };

    currentDriveObject = initialData;

    fs.writeFileSync(filePath, JSON.stringify(initialData, null, 4), "utf8");
    console.log(`Log file created: ${filePath}`);
};

/**
 * Updates the existing log file with the current drive object.
 */
const updateDriveDataLog = () => {
    const filePath = path.join(logsDir, currentDriveLogName);
    fs.writeFileSync(filePath, JSON.stringify(currentDriveObject, null, 4), "utf8");
    console.log(`Log file updated: ${filePath}`);
};

// Create the log file when this module is first imported
createDriveLogFile();

// Export functions
module.exports = {
    /**
     * Retrieves the current log file name.
     * @returns {string} Current log file name.
     */
    getCurrentFileName: () => currentDriveLogName,

    /**
     * Retrieves the absolute path to the current log file.
     * @returns {string} Absolute path to the current log file.
     */
    getCurrentFilePath: () => path.join(logsDir, currentDriveLogName),
    updateDriveDataLog,
    currentDriveObject,
};
