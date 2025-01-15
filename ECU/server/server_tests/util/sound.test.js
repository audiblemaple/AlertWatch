/**
 * @file audioProcessing.test.js
 * @description Unit tests for audioProcessing.js
 */

const path = require("path");
const { spawn } = require("child_process");

// 1. Mock child_process spawn & execFile so we don’t run real commands
jest.mock("child_process", () => ({
  spawn: jest.fn(),
  execFile: jest.fn()
}));

// 2. Mock the `ffmpeg` module so it doesn’t do real recording
jest.mock("fluent-ffmpeg", () => {
  const mockFFmpeg = () => ({
    input: jest.fn().mockReturnThis(),
    inputFormat: jest.fn().mockReturnThis(),
    audioFrequency: jest.fn().mockReturnThis(),
    audioChannels: jest.fn().mockReturnThis(),
    audioCodec: jest.fn().mockReturnThis(),
    format: jest.fn().mockReturnThis(),
    duration: jest.fn().mockReturnThis(),
    on: function (eventName, cb) {
      if (eventName === "end") {
        // We'll call end callback immediately, or you can store it for later
        this._onEnd = cb;
      }
      if (eventName === "error") {
        this._onError = cb;
      }
      return this;
    },
    save: function () {
      // By default, call end right away
      if (this._onEnd) this._onEnd();
    }
  });
  // Return this mock as the default export
  mockFFmpeg.setFfmpegPath = jest.fn();
  return mockFFmpeg;
});

// 3. Import the module under test, AFTER mocks
const { playSound, askForUserConfirmation } = require("../../util/sound");

// 4. Also mock or define user_status from the global if needed
//    If your real global.js exports user_status, you can do:
jest.mock("../../util/global", () => ({
  user_status: {
    userResponded: 1,
    noResponse: 2,
    failedToParse: 3
  },
  locks: {
    alert_lock: false
  }
}));

// If your code uses phrases from const, mock them or define them too:
jest.mock("../../util/const", () => ({
  confirmationPhrases: ["yes i am awake", "i'm good", "all good"],
  noResponsePhrases: ["no", "not responding"]
}));

describe("audioProcessing.js", () => {
  let spawnMock, execFileMock;

  beforeEach(() => {
    // 5. Reset mocks
    jest.clearAllMocks();
    spawnMock = require("child_process").spawn;
    execFileMock = require("child_process").execFile;
  });

  // ---------------------------------------------------------------------------
  // playSound
  // ---------------------------------------------------------------------------
  describe("playSound()", () => {
    it("should spawn 'aplay' with the correct file and resolve on exit code 0", async () => {
      // 1. Mock spawn() to return a fake child process
      const fakeChild = {
        on: jest.fn((event, cb) => {
          if (event === "close") {
            // simulate success exit code
            setTimeout(() => cb(0), 10);
          }
          if (event === "error") {
            // do nothing here
          }
        })
      };
      spawnMock.mockReturnValue(fakeChild);

      const testFile = "/fake/sound.wav";
      await expect(playSound(testFile)).resolves.toBeUndefined();

      expect(spawnMock).toHaveBeenCalledWith("aplay", [testFile]);
      expect(fakeChild.on).toHaveBeenCalledWith("close", expect.any(Function));
    });

    it("should reject if exit code != 0", async () => {
      const fakeChild = {
        on: jest.fn((event, cb) => {
          if (event === "close") {
            // simulate failure exit code
            setTimeout(() => cb(1), 10);
          }
        })
      };
      spawnMock.mockReturnValue(fakeChild);

      await expect(playSound("/fake/sound.wav")).rejects.toThrow(
        /exited with code 1/
      );
    });

    it("should reject if spawn error occurs", async () => {
      const fakeChild = {
        on: jest.fn((event, cb) => {
          if (event === "error") {
            setTimeout(() => cb(new Error("spawn error")), 10);
          }
        })
      };
      spawnMock.mockReturnValue(fakeChild);

      await expect(playSound("/fake/sound.wav")).rejects.toThrow("spawn error");
    });
  });

  // ---------------------------------------------------------------------------
  // askForUserConfirmation
  // ---------------------------------------------------------------------------
  describe("askForUserConfirmation()", () => {
    /**
     * askForUserConfirmation does:
     *  1) recordAudioWithFFmpeg => we mock via fluent-ffmpeg
     *  2) transcribeWithWhisper => calls execFile(whisperCliPath, ...)
     *  3) parseWhisperOutput => transforms the raw lines
     *  4) hasAlertConfirmation => checks phrases vs lines
     *  => returns user_status
     */

    it("should return userResponded if the transcript includes a confirmation phrase", async () => {
      // 1. Mock execFile to simulate a successful transcript
      execFileMock.mockImplementation((cliPath, args, opts) => {
        const fakeChild = {
          stdout: {
            on: (event, cb) => {
              if (event === "data") {
                // simulate some transcript data
                setTimeout(() => cb("[00:00:01] yes i am awake\n"), 5);
              }
            }
          },
          stderr: {
            on: (event, cb) => {}
          },
          on: (event, cb) => {
            if (event === "close") {
              setTimeout(() => cb(0), 15); // success exit code
            }
          }
        };
        return fakeChild;
      });

      const status = await askForUserConfirmation();
      // Should match user_status.userResponded = 1
      expect(status).toBe(1);
    });

    it("should return noResponse if transcript includes a no-response phrase", async () => {
      execFileMock.mockImplementation((cliPath, args, opts) => {
        const fakeChild = {
          stdout: {
            on: (event, cb) => {
              if (event === "data") {
                setTimeout(() => cb("[00:00:01] no\n"), 5);
              }
            }
          },
          stderr: { on: () => {} },
          on: (event, cb) => {
            if (event === "close") {
              setTimeout(() => cb(0), 15);
            }
          }
        };
        return fakeChild;
      });

      const status = await askForUserConfirmation();
      // user_status.noResponse = 2
      expect(status).toBe(2);
    });

    it("should return failedToParse if transcript has none of the known phrases", async () => {
      execFileMock.mockImplementation((cliPath, args, opts) => {
        const fakeChild = {
          stdout: {
            on: (event, cb) => {
              if (event === "data") {
                setTimeout(() => cb("[00:00:01] random gibberish\n"), 5);
              }
            }
          },
          stderr: { on: () => {} },
          on: (event, cb) => {
            if (event === "close") {
              setTimeout(() => cb(0), 15);
            }
          }
        };
        return fakeChild;
      });

      const status = await askForUserConfirmation();
      // user_status.failedToParse = 3
      expect(status).toBe(3);
    });

    it("should reject if whisper-cli fails (exit code != 0)", async () => {
      execFileMock.mockImplementation((cliPath, args, opts) => {
        const fakeChild = {
          stdout: { on: () => {} },
          stderr: { on: () => {} },
          on: (event, cb) => {
            if (event === "close") {
              setTimeout(() => cb(1), 15); // fail
            }
          }
        };
        return fakeChild;
      });

      await expect(askForUserConfirmation()).rejects.toThrow(
        /whisper-cli failed with exit code 1/
      );
    });
  });
});
