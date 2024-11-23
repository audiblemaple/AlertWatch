const {DEBUG} = process.env;


function printToConsole(message){
    if (DEBUG === "1")
        console.log(message);
}

module.exports={printToConsole}