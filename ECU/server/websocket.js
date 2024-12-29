const path = require("path");
const {maxSpeed} = process.env;
const WebSocket = require("ws");
const {playSound, askForUserConfirmation} = require("./util/sound");
const {printToConsole} = require("./util/util");
const {startSpeedBroadcast} = require("./util/carManager");
const {currentDriveObject, updateDriveDataLog} = require("./util/driveLogManager");

const {user_status, locks} = require("./util/const");
let detectionUnitData = undefined;

// Function to initialize the WebSocket server
function initWebSocket(server) {
    const wss = new WebSocket.Server({server});

    // Event listener for new connections
    wss.on("connection", (ws) => {

        // Send a welcome message to the new client
        sendWelcomeMessage(ws);

        // Handle messages from the client
        ws.on("message", (message) => {
            handleClientMessage(ws, message, wss).then(r => printToConsole("Returned from alert message"));
        });

        // Handle client disconnection
        ws.on("close", () => {
           // console.log("WebSocket client disconnected");
        });
    });

    // Start broadcasting speed updates using the WebSocket server instance
    startSpeedBroadcast(wss);
    return wss;
}

// Function to send a welcome message
function sendWelcomeMessage(ws) {
    let tickList = [];
    for (let i = 0; i <= maxSpeed; i += 20)
        tickList.push(i);

    const welcomeMessage = {
        type: "welcome",
        gaugeConf: {
            tickList: tickList,
            redline: {
                minVal: 160,
                maxVal: 220
            },
        },
        detectionUnitData: detectionUnitData
    };
    ws.send(JSON.stringify(welcomeMessage));
}

// Function to broadcast a message to all clients
function broadcast(wss, data) {
    const message = JSON.stringify(data);
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

// Function to handle messages from clients
async function handleClientMessage(ws, message, wss) {
    try {
        const data = JSON.parse(message);
        const response = {
            type: null,
            data: data,
            event: null
        };

        const {type, msgData} = data;

        printToConsole("Received type: " + type)

        switch (type) {
            case "detection_feed":
                // Decode the image frame (base64 to buffer)
                if (!("face_tensors" in msgData) || !("frame" in msgData)) return;

                const {frame, face_tensors, face_landmark_tensors, commands} = msgData;

                if (typeof commands !== 'undefined')
                    detectionUnitData = commands

                // Broadcast detection data to all connected clients
                broadcast(wss, {
                    type: "detection_feed",
                    msgData: {
                        frame: frame,
                        ...(typeof face_tensors !== 'undefined' && { face_tensors }),
                        ...(typeof face_landmark_tensors !== 'undefined' && { face_landmark_tensors }),
                    }
                });
                break;

            case "accelerate":
                printToConsole(msgData);
                break;

            case "decelerate":
                printToConsole(msgData);
                break;

            case "cruise":
                printToConsole(msgData);
                break;

            case "alert": {
                // ─────────────────────────────────────────────────────────────────────────────
                // 1. Check if alert messages are locked (simple semaphore logic)
                // ─────────────────────────────────────────────────────────────────────────────
                if (locks.alert_lock) {
                    printToConsole("alert messages are locked");
                    return;
                }
                locks.alert_lock = true;

                // ─────────────────────────────────────────────────────────────────────────────
                // 2. Destructure event and set up sound paths
                // ─────────────────────────────────────────────────────────────────────────────
                const { event } = data;

                const sounds = {
                    takeABreak:         path.join(__dirname, 'assets/sounds', "takeABreak.wav"),
                    attentionTest:      path.join(__dirname, 'assets/sounds', 'attentionTest.wav'),
                    failedToParse:      path.join(__dirname, 'assets/sounds', 'failedToParse.wav'),
                    noResponse:         path.join(__dirname, 'assets/sounds', 'noResponse.wav'),
                    decelerateWarning:  path.join(__dirname, 'assets/sounds', 'decelerateWarning.wav'),
                    decelerating:       path.join(__dirname, 'assets/sounds', 'decelerating.wav'),
                };

                // ─────────────────────────────────────────────────────────────────────────────
                // 3. If we’ve triggered a “medium alert” before, play attentionTest
                //    and loop until user responds or alert is escalated
                // ─────────────────────────────────────────────────────────────────────────────
                if (currentDriveObject.medium_alert_num > 0) {
                    await playSound(sounds.attentionTest);

                    try {
                        // 3A. Wait for user confirmation in a loop
                        while (true) {
                            const isConfirmedAlert = await askForUserConfirmation();

                            if (isConfirmedAlert === user_status.userResponded) {
                                // User confirmed alert
                                printToConsole("User has confirmed being alert.");
                                await playSound(sounds.takeABreak);
                                currentDriveObject.consecutive_alert_num = 0;
                                break;
                            } else if (isConfirmedAlert === user_status.noResponse) {
                                // User did not confirm → increment counter
                                currentDriveObject.consecutive_alert_num += 1;
                                if (currentDriveObject.consecutive_alert_num >= 1) {
                                    break;
                                }
                                printToConsole("User did not confirm being alert.");
                                await playSound(sounds.noResponse);
                            } else if (isConfirmedAlert === user_status.failedToParse) {
                                // Retry if parse failed
                                await playSound(sounds.failedToParse);
                            }
                        }

                        // 3B. If user never responded, escalate: play decelerateWarning
                        if (currentDriveObject.consecutive_alert_num >= 1) {
                            printToConsole("debug 5");
                            currentDriveObject.consecutive_alert_num = 0;

                            await playSound(sounds.decelerateWarning);
                            const isConfirmedAlert = await askForUserConfirmation();

                            if (isConfirmedAlert === user_status.userResponded) {
                                await playSound(sounds.takeABreak);
                                currentDriveObject.consecutive_alert_num = 0;
                            } else if (isConfirmedAlert === user_status.failedToParse) {
                                // Should ideally loop until response is valid
                                await playSound(sounds.failedToParse);
                            } else {
                                await playSound(sounds.decelerating);
                            }
                        }

                    } catch (err) {
                        // In case anything goes wrong
                        console.error("Error:", err.message);
                        locks.alert_lock = true;
                    }

                    break; // End of "alert" logic when medium_alert_num > 0
                }
                    // ─────────────────────────────────────────────────────────────────────────────
                    // 4. If this is the first time medium_alert_num = 0, just play "takeABreak"
                    //    and increment the alert count
                // ─────────────────────────────────────────────────────────────────────────────
                else {
                    await playSound(sounds.takeABreak);
                    currentDriveObject.medium_alert_num += 1;
                }

                break; // End of case "alert"
            }



            // case "alert":
            //     // Some simple "semaphore" logic
            //     if (locks.alert_lock){
            //         printToConsole("alert messages are locked");
            //         return;
            //     }
            //     locks.alert_lock = true;
            //
            //     const {event} = data
            //
            //     let takeABreak       = path.join(__dirname, 'assets/sounds', "takeABreak.wav");
            //     let attentionTest    = path.join(__dirname, 'assets/sounds', 'attentionTest.wav');
            //     let failedToParse    = path.join(__dirname, 'assets/sounds', 'failedToParse.wav');
            //     let noResponse       = path.join(__dirname, 'assets/sounds', 'noResponse.wav');
            //     let decelerateWarning= path.join(__dirname, 'assets/sounds', 'decelerateWarning.wav');
            //     let decelerating     = path.join(__dirname, 'assets/sounds', 'decelerating.wav');
            //
            //     if (currentDriveObject.medium_alert_num > 0) {
            //         await playSound(attentionTest);
            //
            //         try {
            //             while ( true ){
            //                 // check user response
            //                 const isConfirmedAlert = await askForUserConfirmation();
            //
            //                 if (isConfirmedAlert === user_status.user_responded) {
            //                     printToConsole("User has confirmed being alert.");
            //                     await playSound(takeABreak);
            //                     currentDriveObject.consecutive_alert_num = 0;
            //                     break;
            //                 } else if (isConfirmedAlert === user_status.no_response) {
            //                     currentDriveObject.consecutive_alert_num += 1;
            //                     if (currentDriveObject.consecutive_alert_num >= 1)
            //                         break;
            //
            //                     printToConsole("User did not confirm being alert.");
            //                     await playSound(noResponse);
            //                 } else if (isConfirmedAlert === user_status.failed_to_parse)
            //                     await playSound(failedToParse);
            //             }
            //
            //             if (currentDriveObject.consecutive_alert_num >= 1){
            //                 printToConsole("debug 5")
            //
            //                 currentDriveObject.consecutive_alert_num = 0;
            //                 await playSound(decelerateWarning);
            //                 const isConfirmedAlert = await askForUserConfirmation();
            //                  if (isConfirmedAlert === user_status.user_responded) {
            //                      await playSound(takeABreak);
            //                      currentDriveObject.consecutive_alert_num = 0;
            //                  } else if (isConfirmedAlert === user_status.failed_to_parse)
            //                      await playSound(failedToParse); // should also  be like a while loop until we get a response.
            //                  else
            //                      await playSound(decelerating);
            //
            //             }
            //         } catch (err) {
            //             console.error("Error:", err.message);
            //             locks.alert_lock = true;
            //         }
            //         break;
            //     } else {
            //         await playSound(takeABreak);
            //         currentDriveObject.medium_alert_num += 1;
            //     }
            //     break;

            default:
                printToConsole(`unknown type: ${data}`)
                response.type = "unknown";
                locks.alert_lock = false;

        }
        locks.alert_lock = false;

        ws.send(JSON.stringify(response));
    } catch (error) {
        console.error("Invalid JSON received:", error);
    }
}

// Export the initialization function and other WebSocket functions if needed
module.exports = {initWebSocket, broadcast};
