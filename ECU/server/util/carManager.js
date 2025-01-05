require("dotenv").config();
const {maxSpeed, updateFreq} = process.env;
const WebSocket = require("ws");
const {units} = require("./const");
const {carState} = require("./global");
const {getRandomInt} = require("./random");

let speed = 0;

// Function to generate a random speed update
function updateSpeed() {
    if (carState.accelerating)
        accelerateCar();
    if (carState.decelerating || speed >= 120)
        decelerateCar();
    if (carState.cruising     || speed >= 100)
        cruiseDrive();
    // if (carState.stopped)
    //     console.log("car stopped"); // TODO: add logic of constant beeping
}

function logStatus(){
    console.log("accelerating: ", carState.accelerating)
    console.log("decelerating: ", carState.decelerating)
    console.log("stopped: "     , carState.stopped)
    console.log("cruising: "        , carState.cruising)
    console.log("speed: ", speed)
    console.log("\n")
}

// Function to start broadcasting speed updates
function startSpeedBroadcast(wss) {
    setInterval(() => {
        updateSpeed();
        // logStatus();
        broadcastSpeed(wss, speed);
    }, updateFreq * units.second);
}

// Function to broadcast speed to all connected WebSocket clients
function broadcastSpeed(wss, speed) {
    if (carState.stopped)
        return;

    const message = JSON.stringify({
        type: "speed",
        msgData: speed
    });
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

function accelerateCar(){
    carState.accelerating = true;
    carState.decelerating = false;
    carState.stopped      = false;
    carState.cruising     = false;
    speed += Math.floor(9 * ((7 + getRandomInt(5)) / 10));
}

function decelerateCar(){
    carState.accelerating = false;
    carState.decelerating = true;
    carState.stopped      = false;
    carState.cruising       = false;
    if (speed <= 0){
        carState.stopped = true;
        carState.decelerating = false;
        speed = 0;
        return
    }
    speed += Math.floor(-9 * ((7 + getRandomInt(3)) / 10));
}

function cruiseDrive(){
    carState.accelerating = false;
    carState.decelerating = false;
    carState.stopped      = false;
    carState.cruising       = true;

    getRandomInt(101) >= 50 ? speed += getRandomInt(5) :  speed -= getRandomInt(3);
}



module.exports = { startSpeedBroadcast, maxSpeed, accelerateCar, decelerateCar, cruiseDrive};
