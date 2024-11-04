const fs = require("fs");
const path = require("path");

let currentDriveLogName = "";
let currentDriveObject = {}

// Function to generate a timestamped filename
// Function to generate a custom timestamped filename
const generateDriveFileName = () => {
    const date = new Date();
    const seconds = String(date.getSeconds()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const month = String(date.getMonth() + 1).padStart(2, "0"); // Month is zero-based
    const year = date.getFullYear();

    return `../logs/Drive-${seconds}-${minutes}-${hours}T${day}-${month}-${year}.json`;
};

// Function to create a log file with a timestamped name
const createDriveLogFile = () => {
    currentDriveLogName = generateDriveFileName();
    const filePath = path.join(__dirname, currentDriveLogName);
    const initialData = {
        timestamp: new Date().toISOString(),
        alert_count: 0,
        mild_alert_num: 0,
        medium_alert_num: 0,
        high_alert_num: 0,
    };

    currentDriveObject = initialData;

    fs.writeFileSync(filePath, JSON.stringify(initialData, null, 4), "utf8");
    console.log(`Log file created: ${currentDriveLogName}`);
};

// Function to update log file with new data, adding and updating fields as necessary
const updateDriveDataLog = () => {
    // Write the updated object back to the file
    const filePath = path.join(__dirname, currentDriveLogName);
    fs.writeFileSync(filePath, JSON.stringify(currentDriveObject, null, 4), "utf8");
    console.log(`Log file updated: ${currentDriveLogName}`);
};

// Create the log file when this module is first imported
createDriveLogFile();

// Export the current filename and a helper function to get the file path
module.exports = {
    getCurrentFileName: () => currentDriveLogName,
    getCurrentFilePath: () => path.join(__dirname, currentDriveLogName),
    updateDriveDataLog,
    currentDriveObject,
};
