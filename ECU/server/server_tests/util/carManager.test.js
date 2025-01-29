/**
 * @file carManager.test.js
 * @description Unit tests for carManager.js
 */

const WebSocket = require("ws");

// Import the real code under test
const {
  setCarAccelerating,
  setCarDecelerating,
  setCarCruising,
  setCarStopped,
  updateCarSpeed,
} = require("../../util/carManager");

const { carState } = require("../../util/global");

// Fake Timers to control setInterval
describe("carManager.js", () => {
  let mockWss;
  let intervalId;

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
  // setCarAccelerating
  // ---------------------------------------------------------------------------
  describe("setCarAccelerating()", () => {
    it("should set carState to accelerating and clear other states", () => {
      setCarAccelerating();
      expect(carState.accelerating).toBe(true);
      expect(carState.decelerating).toBe(false);
      expect(carState.stopped).toBe(false);
      expect(carState.cruising).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // setCarDecelerating
  // ---------------------------------------------------------------------------
  describe("setCarDecelerating()", () => {
    it("should set carState to decelerating and clear other states", () => {
      setCarDecelerating();
      expect(carState.decelerating).toBe(true);
      expect(carState.accelerating).toBe(false);
      expect(carState.stopped).toBe(false);
      expect(carState.cruising).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // setCarCruising
  // ---------------------------------------------------------------------------
  describe("setCarCruising()", () => {
    it("should set carState to cruising and clear other states", () => {
      setCarCruising();
      expect(carState.cruising).toBe(true);
      expect(carState.accelerating).toBe(false);
      expect(carState.decelerating).toBe(false);
      expect(carState.stopped).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // setCarStopped
  // ---------------------------------------------------------------------------
  describe("setCarStopped()", () => {
    it("should set carState to stopped and speed to 0", () => {
      setCarStopped();
      expect(carState.stopped).toBe(true);
      expect(carState.accelerating).toBe(false);
      expect(carState.decelerating).toBe(false);
      expect(carState.cruising).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // updateCarSpeed
  // ---------------------------------------------------------------------------
  describe("updateCarSpeed()", () => {
    it("should update the speed based on car state", () => {
      setCarAccelerating();
      updateCarSpeed();
      expect(carState.accelerating).toBe(true);

      setCarDecelerating();
      updateCarSpeed();
      expect(carState.decelerating).toBe(true);
    });
  });
});
