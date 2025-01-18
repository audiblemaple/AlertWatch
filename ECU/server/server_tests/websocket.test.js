/**
 * @file websocket.test.js
 * @description Unit tests for websocket.js
 */

jest.setTimeout(10000); // Increase test timeout if needed

// 1. Set process.env.maxSpeed BEFORE requiring the code that uses it
process.env.maxSpeed = "220";

const http = require("http");
const WebSocket = require("ws");

// 2. Mock dependencies
jest.mock("../util/sound", () => ({
  playSound: jest.fn(),
  askForUserConfirmation: jest.fn(),
}));

jest.mock("../util/util", () => ({
  printToConsole: jest.fn(),
  getSystemData: jest.fn().mockReturnValue({
    os: "Test OS",
    version: "1.0.0",
  }),
}));

jest.mock("../util/carManager", () => ({
  startSpeedBroadcast: jest.fn(),
  decelerateCar: jest.fn(),
  accelerateCar: jest.fn(),
}));

jest.mock("../util/driveLogManager", () => ({
  currentDriveObject: {
    medium_alert_num: 0,
    consecutive_alert_num: 0,
  },
  updateDriveDataLog: jest.fn(),
}));

jest.mock("../util/global", () => ({
  user_status: {
    userResponded: "userResponded",
    noResponse: "noResponse",
    failedToParse: "failedToParse",
  },
  locks: {
    alert_lock: false,
  },
  sounds: {
    takeABreak: "takeABreak.wav",
    attentionTest: "attentionTest.wav",
    failedToParse: "failedToParse.wav",
    noResponse: "noResponse.wav",
    decelerating: "decelerating.wav",
  },
}));

// 3. Now import the real module under test
//    Adjust the path if your file is named differently or in another folder
const { initWebSocket, broadcast } = require("../websocket");

// 4. Get references to mocked modules/functions
const { playSound, askForUserConfirmation } = require("../util/sound");
const { printToConsole, getSystemData } = require("../util/util");
const { startSpeedBroadcast, decelerateCar, accelerateCar } = require("../util/carManager");
const { currentDriveObject } = require("../util/driveLogManager");
const { locks, user_status, sounds } = require("../util/global");

describe("websocketServer.js", () => {
  let server; // real HTTP server
  let wss;    // WebSocket.Server instance
  let port;   // assigned port

  beforeAll((done) => {
    // 1. Create a real HTTP server
    server = http.createServer();

    // 2. Listen on a random port (0) to avoid collisions
    server.listen(0, () => {
      port = server.address().port;
      done();
    });
  });

  afterAll((done) => {
    // 3. Clean up the HTTP server after the test suite
    server.close(done);
  });

  beforeEach(() => {
    // Clear mocks before each test
    jest.clearAllMocks();

    // Reset lock states and drive data
    locks.alert_lock = false;
    currentDriveObject.medium_alert_num = 0;
    currentDriveObject.consecutive_alert_num = 0;

    // 4. Create a fresh WebSocket.Server for each test
    wss = initWebSocket(server);
  });

  afterEach(() => {
    // Close the WebSocket server if it's still open
    if (wss && wss.close) {
      wss.close();
    }
  });

  // ---------------------------------------------------------------------------
  // Test: initWebSocket
  // ---------------------------------------------------------------------------
  describe("initWebSocket(server)", () => {
    it("should initialize a WebSocket server and call startSpeedBroadcast", () => {
      // At this point, wss is already created in beforeEach
      expect(wss).toBeInstanceOf(WebSocket.Server);
      expect(startSpeedBroadcast).toHaveBeenCalledWith(wss);
    });
  });

  // ---------------------------------------------------------------------------
  // Test: broadcast
  // ---------------------------------------------------------------------------
  describe("broadcast(wss, data)", () => {
    it("should send data to all connected clients", (done) => {
      const client = new WebSocket(`ws://localhost:${port}`);

      // 1. Handle messages from the server
      client.on("message", (rawMsg) => {
        const msg = JSON.parse(rawMsg);

        if (msg.type === "welcome") {
          // This is the first message the server sends on connection
          // We'll ignore or verify it, then broadcast the message we want.
          broadcast(wss, { type: "test_broadcast" });
        } else {
          // This should be our broadcast
          expect(msg).toEqual({ type: "test_broadcast" });
          client.close();
          done();
        }
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Test: handleClientMessage (via sending messages)
  // ---------------------------------------------------------------------------
  describe("handleClientMessage()", () => {
    it("should lock alerts when receiving an 'alert' message, then unlock after playSound", (done) => {
      // Ensure the mock resolves quickly
      playSound.mockResolvedValueOnce();

      const client = new WebSocket(`ws://localhost:${port}`);

      client.on("open", () => {
        // 1. Wait for the initial welcome
        client.once("message", (welcomeMsg) => {
          // 2. Now send the alert
          client.send(JSON.stringify({ type: "alert", msgData: "low_average_ear" }));
        });
      });

      client.on("message", (msg) => {
        const data = JSON.parse(msg);
        // If it's the response from the alert handling...
        if (data.type !== "welcome") {
          // 3. Check whether `playSound` was called, etc.
          setTimeout(() => {
            expect(playSound).toHaveBeenCalledWith(sounds.takeABreak);
            expect(locks.alert_lock).toBe(false);

            client.close();
            done();
          }, 500);
        }
      });
    });

    // it("should handle detection_feed messages and broadcast them to connected clients", (done) => {
    //   const client1 = new WebSocket(`ws://localhost:${port}`);
    //
    //   client1.on("open", () => {
    //     // Ignore initial welcome
    //     client1.once("message", () => {
    //       // Then send detection_feed
    //       const detectionPayload = {
    //         type: "detection_feed",
    //         msgData: {
    //           frame: "base64EncodedFrameData",
    //           face_tensors: [1, 2, 3],
    //           face_landmark_tensors: [4, 5, 6],
    //           commands: { someCommand: "test" },
    //         },
    //       };
    //       client1.send(JSON.stringify(detectionPayload));
    //     });
    //   });

      // Connect second client to verify broadcast
      // const client2 = new WebSocket(`ws://localhost:${port}`);
      //
      // let gotWelcome = false;
      // let gotDetectionFeed = false;
      //
      // client2.on("message", (rawMsg) => {
      //   const msg = JSON.parse(rawMsg);
      //
      //   if (msg.type === "welcome") {
      //     gotWelcome = true;
      //   } else if (msg.type === "detection_feed") {
      //     gotDetectionFeed = true;
      //     expect(msg.msgData.frame).toBe("base64EncodedFrameData");
      //     expect(msg.msgData.face_tensors).toEqual([1, 2, 3]);
      //     expect(msg.msgData.face_landmark_tensors).toEqual([4, 5, 6]);
      //     client1.close();
      //     client2.close();
      //     done();
      //   }
      // });

    // });
  });

  // ---------------------------------------------------------------------------
  // Test: sendWelcomeMessage() (indirectly tested on connection)
  // ---------------------------------------------------------------------------
  describe("sendWelcomeMessage()", () => {
    it("should send a welcome message upon client connection", (done) => {
      const client = new WebSocket(`ws://localhost:${port}`);

      client.on("message", (data) => {
        const parsed = JSON.parse(data);

        if (parsed.type === "welcome") {
          // The gaugeConf.tickList should contain increments of 20 up to maxSpeed=220
          // For example: [0, 20, 40, 60, ... , 220]
          expect(parsed.gaugeConf.tickList).toContain(0);
          expect(parsed.gaugeConf.tickList).toContain(220);

          // System data is from mocked getSystemData
          expect(parsed.systemData).toHaveProperty("os", "Test OS");
          expect(parsed.systemData).toHaveProperty("version", "1.0.0");

          client.close();
          done();
        }
      });
    });
  });
});
