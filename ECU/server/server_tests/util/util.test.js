/**
 * @file util.test.js
 * @description Unit tests for util.js
 */

// 1. Mock child_process for getSystemData
const { execSync } = require("child_process");
jest.mock("child_process", () => ({
  execSync: jest.fn()
}));

// 2. Reset environment & mocks before each test
beforeEach(() => {
  process.env.DEBUG = "0"; // default to no logging
  jest.clearAllMocks();
});

// 3. Import your module AFTER the mocks
const {
  printToConsole,
  getSystemData,
  getRandomInt
} = require("../../util/util");

describe("utility.js", () => {

  // ---------------------------------------------------------------------------
  // printToConsole
  // ---------------------------------------------------------------------------
  describe("printToConsole()", () => {
    let consoleSpy;

    beforeAll(() => {
      // Spy on console.log
      consoleSpy = jest.spyOn(console, "log").mockImplementation(() => {});
    });

    afterAll(() => {
      consoleSpy.mockRestore();
    });

    it("should NOT log if DEBUG != '1'", () => {
      process.env.DEBUG = "0";
      printToConsole("Should not see this");
      expect(consoleSpy).not.toHaveBeenCalled();
    });

    it("should log if DEBUG == '1'", () => {
      process.env.DEBUG = "1";
      printToConsole("Hello again");
      expect(consoleSpy).toHaveBeenCalledWith("Hello again");
    });
  });

  // ---------------------------------------------------------------------------
  // getSystemData
  // ---------------------------------------------------------------------------
  describe("getSystemData()", () => {
    // it("should return parsed system data when commands succeed", () => {
    //   // Mock execSync for CPU, memory, hailo
    //   execSync.mockImplementation((command) => {
    //     if (command.includes("lscpu")) {
    //       return Buffer.from("Fake CPU Model");
    //     }
    //     if (command.startsWith("free -h")) {
    //       return Buffer.from([
    //         "              total        used        free    shared  buff/cache   available",
    //         "Mem:         16G         10G         4G       0.1G    1G           5G",
    //         "Swap:        2G          0G          2G"
    //       ].join("\n"));
    //     }
    //     if (command.startsWith("hailortcli")) {
    //       return Buffer.from([
    //         "Device ID: 1234",
    //         "Firmware: v2.3.4",
    //         "Status: Active"
    //       ].join("\n"));
    //     }
    //   });
    //
    //   const data = getSystemData();
    //   expect(data.cpuModel).toBe("Fake CPU Model");
    //   expect(data.memory).toEqual(
    //     expect.objectContaining({
    //       total: "16G",
    //       used: "10G",
    //       free: "4G",
    //       available: "5G"
    //     })
    //   );
    //   expect(data.hailoInfo).toEqual({
    //     DeviceID: "1234",
    //     Firmware: "v2.3.4",
    //     Status: "Active"
    //   });
    // });

    // it("should return fallback data if commands throw an error", () => {
    //   // Make execSync throw
    //   execSync.mockImplementation(() => {
    //     throw new Error("Some error");
    //   });
    //
    //   const data = getSystemData();
    //   expect(data).toEqual({
    //     cpuModel: "Unknown",
    //     memory: {},
    //     hailoInfo: {}
    //   });
    // });
  });

  // ---------------------------------------------------------------------------
  // getRandomInt
  // ---------------------------------------------------------------------------
  describe("getRandomInt()", () => {
    it("should return a random integer < max", () => {
      const max = 5;
      const val = getRandomInt(max);
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThan(max);
    });

    it("should produce multiple different values over many calls", () => {
      const values = new Set();
      for (let i = 0; i < 50; i++) {
        values.add(getRandomInt(100));
      }
      // We expect at least some variety
      expect(values.size).toBeGreaterThan(1);
    });
  });

});
