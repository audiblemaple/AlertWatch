const {DEBUG} = process.env;
const fs = require('fs')
const { execSync } = require('child_process');
function printToConsole(message){
    if (DEBUG === "1")
        console.log(message);
}

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

function getSystemData() {
    try {
        const cpuModel = execSync('lscpu | grep -i "model name" | awk -F: \'{print $2}\'').toString().trim();
        const memory = execSync('free -h').toString();
        const hailoInfo = execSync('hailortcli fw-control identify').toString();

        return {
            cpuModel: cpuModel,
            memory: parseMemoryData(memory),
            hailoInfo: parseHailoData(hailoInfo)
        };
    } catch (error) {
        console.error('Error gathering system data:', error);
        return {
            cpuModel: "Unknown",
            memory: {},
            hailoInfo: {}
        };
    }
}

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

module.exports={printToConsole, getSystemData}