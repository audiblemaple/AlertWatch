/**
 * @file carSpeedManager.js
 * @description Manages car speed states and broadcasts updates to WebSocket clients.
 * @author Lior Jigalo
 * @license MIT
 */

/** Load environment variables from .env file */
require("dotenv").config();

/** Extract environment variables */
const {maxSpeed, updateFreq} = process.env;

/** Import WebSocket library */
const WebSocket = require("ws");

/** Import constants and utility functions */
const {units} = require("./const");
const {carState} = require("./global");
const {getRandomInt} = require("./util");

/** Current speed of the car, starting from 0 */
let speed = 0;

/**
 * Updates the car's speed based on its state.
 */
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

/**
 * Logs the current state of the car to the console.
 */
function logStatus(){
    console.log("accelerating: ", carState.accelerating)
    console.log("decelerating: ", carState.decelerating)
    console.log("stopped: "     , carState.stopped)
    console.log("cruising: "        , carState.cruising)
    console.log("speed: ", speed)
    console.log("\n")
}

/**
 * Starts broadcasting the car's speed to all connected WebSocket clients.
 * @param {WebSocket.Server} wss - The WebSocket server instance.
 */
function startSpeedBroadcast(wss) {
    setInterval(() => {
        updateSpeed();
        // logStatus();
        broadcastSpeed(wss, speed);
    }, updateFreq * units.second);
}

/**
 * Broadcasts the current speed to all connected WebSocket clients.
 * @param {WebSocket.Server} wss - The WebSocket server instance.
 * @param {number} speed - The current speed of the car.
 */
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

/**
 * Accelerates the car and updates its state.
 */
function accelerateCar(){
    carState.accelerating = true;
    carState.decelerating = false;
    carState.stopped      = false;
    carState.cruising     = false;
    speed += Math.floor(9 * ((7 + getRandomInt(5)) / 10));
}

/**
 * Decelerates the car and updates its state.
 */
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

/**
 * Sets the car to cruise mode and updates its state.
 */
function cruiseDrive(){
    carState.accelerating = false;
    carState.decelerating = false;
    carState.stopped      = false;
    carState.cruising       = true;

    getRandomInt(101) >= 50 ? speed += getRandomInt(5) :  speed -= getRandomInt(3);
}

/** Export functions */
module.exports = {
    startSpeedBroadcast,
    maxSpeed,
    accelerateCar,
    decelerateCar,
    cruiseDrive,
};