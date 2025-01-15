/**
 * @file carManager.test.js
 * @description Unit tests for carManager.js
 */

const WebSocket = require("ws");

// 1. Mock or set environment variables
process.env.maxSpeed = "180";
process.env.updateFreq = "0.1"; // a tenth of a second for quicker tests

// 2. Mock or set your const and global modules as needed
jest.mock("../../util/const", () => ({
  units: {
    second: 1, // so updateFreq * 1 = 0.1s
  },
}));

// 3. Import the real code under test
const {
  startSpeedBroadcast,
  accelerateCar,
  decelerateCar,
  cruiseDrive,
} = require("../../util/carManager");

// 4. Also import your global carState
const { carState } = require("../../util/global");

// 5. Use Fake Timers so we can control setInterval
describe("carManager.js", () => {
  let mockWss;
  let intervalId; // We'll store the returned ID so we can clear it

  beforeAll(() => {
    jest.useFakeTimers();
  });

  beforeEach(() => {
    // Reset carState fields before each test
    carState.accelerating = false;
    carState.decelerating = false;
    carState.stopped = false;
    carState.cruising = false;

    // Create a mock WSS with a mock client
    mockWss = {
      clients: new Set(),
    };

    // Add a mock client that can receive messages
    const mockClient = {
      readyState: WebSocket.OPEN,
      send: jest.fn(),
    };
    mockWss.clients.add(mockClient);

    intervalId = null; // reset
  });

  afterEach(() => {
    // Clear the interval if it was set
    if (intervalId) {
      clearInterval(intervalId);
    }
    jest.clearAllMocks();
  });

  // ---------------------------------------------------------------------------
  // accelerateCar
  // ---------------------------------------------------------------------------
  describe("accelerateCar()", () => {
    it("should set carState to accelerating and increase speed", () => {
      accelerateCar();
      expect(carState.accelerating).toBe(true);
      expect(carState.decelerating).toBe(false);
      expect(carState.stopped).toBe(false);
      expect(carState.cruising).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // decelerateCar
  // ---------------------------------------------------------------------------
  describe("decelerateCar()", () => {
    it("should set carState to decelerating if speed > 0, or stopped if speed <= 0", () => {
      accelerateCar(); // speed > 0
      decelerateCar();
      expect(carState.decelerating).toBe(true);

      // Keep decelerating until it's stopped
      for (let i = 0; i < 20; i++) {
        decelerateCar();
        if (carState.stopped) break;
      }
      expect(carState.stopped).toBe(true);
      expect(carState.decelerating).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // cruiseDrive
  // ---------------------------------------------------------------------------
  describe("cruiseDrive()", () => {
    it("should set carState to cruising", () => {
      cruiseDrive();
      expect(carState.cruising).toBe(true);
      expect(carState.accelerating).toBe(false);
      expect(carState.decelerating).toBe(false);
      expect(carState.stopped).toBe(false);
    });
  });
});
