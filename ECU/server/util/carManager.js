require("dotenv").config();
const {maxSpeed, updateFreq} = process.env;
const WebSocket = require("ws");
const {units} = require("./const");
const {getRandomInt} = require("./random");

let speed = 0;

let accelerating = true;
let decelerating = false;
let stopped      = false;
let cruise       = false;

// Function to generate a random speed update
function updateSpeed() {
    if (accelerating)
        accelerateCar();
    if (decelerating || speed >= 135)
        decelerateCar();
    if (cruise       || speed >= 120)
        cruiseDrive();
}

function logStatus(){
    console.log("accelerating: ", accelerating)
    console.log("decelerating: ", decelerating)
    console.log("stopped: "     , stopped)
    console.log("cruise: "        , cruise)
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
    accelerating = true;
    decelerating = false;
    stopped      = false;
    cruise       = false;
    speed += Math.floor(9 * ((7 + getRandomInt(5)) / 10));
}

function decelerateCar(){
    accelerating = false;
    decelerating = true;
    stopped      = false;
    cruise       = false;
    speed += Math.floor(-9 * ((7 + getRandomInt(5)) / 10));
}

function cruiseDrive(){
    accelerating = false;
    decelerating = false;
    stopped      = false;
    cruise       = true;

    getRandomInt(101) >= 50 ? speed += getRandomInt(5) :  speed -= getRandomInt(5);
}

module.exports = { startSpeedBroadcast, maxSpeed };
