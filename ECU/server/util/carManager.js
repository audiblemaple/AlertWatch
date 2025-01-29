/**
 * @file carSpeedManager.js
 * @description Manages car speed states and broadcasts updates to WebSocket clients.
 * @author ...
 * @license MIT
 */

/** Load environment variables from .env file */
require("dotenv").config();

/** Extract environment variables */
const { maxSpeed = 120, updateFreq = 1 } = process.env;

/** Import WebSocket library */
const WebSocket = require("ws");

/** Import constants and utility functions */
const { units } = require("./const");
// your getRandomInt function (or any other helper) would go here
const { getRandomInt } = require("./util");
const { carState } = require("./global");


/** Current speed of the car, starting from 0 */
let speed = 0;

/**
 * Resets the car state to all `false`, except the one you want to set.
 * This helps avoid conflicting states (e.g. accelerating AND decelerating at the same time).
 */
function resetCarState() {
    carState.accelerating = false;
    carState.decelerating = false;
    carState.cruising = false;
    carState.stopped = false;
}

/**
 * Sets car state to Accelerating.
 */
function setCarAccelerating() {
    resetCarState();
    carState.accelerating = true;
}

/**
 * Sets car state to Decelerating.
 */
function setCarDecelerating() {
    resetCarState();
    carState.decelerating = true;
}

/**
 * Sets car state to Cruising.
 */
function setCarCruising() {
    resetCarState();
    carState.cruising = true;
}

/**
 * Sets car state to Stopped.
 */
function setCarStopped() {
    resetCarState();
    carState.stopped = true;
    speed = 0; // If stopping, ensure speed is 0
}

/**
 * Updates the car's speed based on its current state.
 * Only call this once per "tick" or interval in your main loop.
 */
function updateCarSpeed() {
    if (carState.stopped)
        // Speed is already forced to 0 when setCarStopped() is called.
        return;

    if (carState.accelerating) {
        if (speed >= 100)
            return setCarCruising();

        // Accelerate in a random-ish way
        const increment = Math.floor(9 * ((7 + getRandomInt(5)) / 10));
        speed = Math.min(speed + increment, maxSpeed);
    }

    if (carState.decelerating) {
        // Decelerate in a random-ish way
        const decrement = Math.floor(9 * ((7 + getRandomInt(3)) / 10));
        speed -= decrement;

        // If we've hit 0 or below, consider that "stopped"
        if (speed <= 0)
            setCarStopped(); // sets speed = 0 and flips to stopped state
    }

    if (carState.cruising) {
        if (speed <= 60)
            return setCarStopped();

        // "Cruise" by randomly adjusting speed up/down a small amount
        // The idea is that cruising won't drastically change speed
        if (getRandomInt(120) >= 50) {
            // Slight speed increase
            speed = Math.min(speed + getRandomInt(3), maxSpeed);
        } else {
            // Slight speed decrease
            speed = Math.max(speed - getRandomInt(5), 0);
        }
    }
}

/**
 * Starts broadcasting the car's speed to all connected WebSocket clients.
 * @param {WebSocket.Server} wss - The WebSocket server instance.
 */
function startSpeedBroadcast(wss) {
    setInterval(() => {
        // Update speed exactly once per loop
        updateCarSpeed();
        broadcastSpeed(wss, speed);
    }, updateFreq * units.second);
}

/**
 * Broadcasts the current speed to all connected WebSocket clients.
 * @param {WebSocket.Server} wss - The WebSocket server instance.
 * @param {number} speed - The current speed of the car.
 */
function broadcastSpeed(wss, speed) {

    const message = JSON.stringify({
        type: "speed",
        msgData: speed,
    });

    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

/** Export functions so other modules can control the car's state and start the loop */
module.exports = {
    // state setters
    updateCarSpeed,
    setCarCruising,
    setCarStopped,
    setCarAccelerating,
    setCarDecelerating,
    startSpeedBroadcast,
    maxSpeed,
};
