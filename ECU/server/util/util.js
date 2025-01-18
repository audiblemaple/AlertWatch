/**
 * @file utility.js
 * @description Utility functions for console logging, file management, system data retrieval, and random number generation.
 * @author Lior Jigalo
 * @license MIT
 */
const fs = require('fs')
const { execSync } = require('child_process');

/**
 * Logs a message to the console if debugging is enabled.
 * @param {string} message - The message to log.
 */
// utility.js
function printToConsole(message){
  // Read process.env.DEBUG each time
  if (process.env.DEBUG === "1") {
    console.log(message);
  }
}

/**
 * Deletes a file at the specified path.
 * @async
 * @param {string} path - The path of the file to delete.
 * @returns {Promise<void>} - Resolves when the file is successfully deleted.
 */
async function removeFile(path) {
    try {
        await fs.unlink(path);
        console.log('File deleted successfully.');
    } catch (err) {
        switch (err.code) {
            case 'ENOENT':
                console.error('File does not exist.');
                break;
            case 'EACCES':
            case 'EPERM':
                console.error('Permission denied. Cannot delete the file.');
                break;
            case 'EBUSY':
                console.error('File is busy or locked.');
                break;
            default:
                console.error('An unexpected error occurred:', err);
        }
    }
}

/**
 * Retrieves system data including CPU model, memory usage, and Hailo device information.
 * @returns {Object} System data containing CPU model, memory usage, and Hailo info.
 */
function getSystemData() {
    try {
        const cpuModel = execSync('lscpu | grep -i "model name" | awk -F: \'{print $2}\'').toString().trim();
        const memory = execSync('free -h').toString();
//        const hailoInfo = execSync('hailortcli fw-control identify').toString();

        return {
            cpuModel: cpuModel,
            memory: parseMemoryData(memory),
//            hailoInfo: parseHailoData(hailoInfo)
        };
    } catch (error) {
        console.error('Error gathering system data:', error);
        return {
            cpuModel: "Unknown",
            memory: {},
//            hailoInfo: {}
        };
    }
}

/**
 * Parses the raw output of Hailo device information into a structured object.
 * @param {string} hailoInfo - Raw output from Hailo CLI.
 * @returns {Object} Parsed Hailo device information.
 */
function parseHailoData(hailoInfo) {
    const hailoLines = hailoInfo.split('\n');
    const data = {};

    hailoLines.forEach(line => {
        const [key, value] = line.split(':').map(str => str.trim());
        if (key && value) {
            data[key.replace(/ /g, '')] = value;
        }
    });

    return data;
}

/**
 * Parses memory usage data from the `free -h` command output, some irrelevant data was excluded to send less data.
 * @param {string} memory - Raw memory data output.
 * @returns {Object} Parsed memory usage information.
 */
function parseMemoryData(memory) {
    const lines = memory.split('\n');
    const [total, used, free, shared, bufferCache, available] = lines[1].split(/\s+/).slice(1);
    const [swapTotal, swapUsed, swapFree] = lines[2].split(/\s+/).slice(1);

    return {
        total,
        used,
        free,
        // shared,
        // bufferCache,
        available,
        // swap: {
        //     total: swapTotal,
        //     used: swapUsed,
        //     free: swapFree
        // }
    };
}

/**
 * Generates a random integer between 0 (inclusive) and the specified maximum value (exclusive).
 * @param {number} max - The upper bound (exclusive) for the random integer.
 * @returns {number} A random integer between 0 and max.
 */
function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

/** Exported utility functions. */
module.exports={printToConsole, getSystemData, getRandomInt}