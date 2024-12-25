const {DEBUG} = process.env;
const fs = require('fs')

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



module.exports={printToConsole, removeFile}