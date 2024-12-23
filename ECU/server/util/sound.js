const { exec, execFile } = require('child_process');
const path = require("path");

// Function to mute all streams
function muteAllStreams() {
  exec("pactl list sink-inputs | grep -oP '(?<=Sink Input #)\\d+'", (err, stdout) => {
    if (err) {
      console.error(`Error listing streams: ${err}`);
      return;
    }
    const streamIds = stdout.trim().split('\n');
    streamIds.forEach(streamId => {
      exec(`pactl set-sink-input-mute ${streamId} 1`);
    });
  });
}

// Function to unmute all streams
function unmuteAllStreams() {
    exec("pactl list sink-inputs | grep -oP '(?<=Sink Input #)\\d+'", (err, stdout) => {
        if (err) {
            console.error(`Error listing streams: ${err}`);
            return;
        }
        const streamIds = stdout.trim().split('\n');
        streamIds.forEach(streamId => {
            exec(`pactl set-sink-input-mute ${streamId} 0`);
        });
    });
}

// Function to play custom sound
function playSound(filePath) {
    muteAllStreams();
    exec(`paplay ${filePath}`, (err) => {
        if (err) {
          console.error(`Error playing sound: ${err}`);
        }
    });
    setTimeout(unmuteAllStreams, 4000);
}




/**
 * Transcribes a WAV file using whisper-cli, capturing stdout/stderr streams.
 * @param {string} audioFilePath - The path to the .wav file
 * @returns {Promise<string>} - The raw transcription output from whisper-cli
 */
async function transcribeWithWhisper(audioFilePath) {
  // Path to the whisper-cli binary
  const whisperCliPath = path.join(__dirname, "../bin", "whisper-cli");
  // Path to the whisper model
  const modelPath = path.join(__dirname, "../models", "ggml-tiny.en.bin");

  // Build arguments
  const args = ["--model", modelPath, "--no-prints", audioFilePath];

  return new Promise((resolve, reject) => {
    // We do NOT pass a callback here, because we want manual streaming.
    const child = execFile(whisperCliPath, args, { shell: false });

    let stdoutData = "";
    let stderrData = "";

    child.stdout.on("data", (data) => {
      stdoutData += data.toString();
    });

    child.stderr.on("data", (data) => {
      stderrData += data.toString();
    });

    child.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new Error(
            `whisper-cli failed with exit code ${code}.\nStderr: ${stderrData}`
          )
        );
      }
      // Return the entire raw transcription output
      resolve(stdoutData.trim());
    });

    // If an error is emitted at the process level, catch it
    child.on("error", (err) => {
      reject(err);
    });
  });
}

// A helper function to parse the raw transcription lines
function parseWhisperOutput(rawOutput) {
  // Each line might look like:
  // [00:00:00.000 --> 00:00:04.000]   alertness check...
  // We'll remove that bracketed part using a regex and just keep the text portion.
  const lines = rawOutput.split("\n").filter(Boolean);

  return lines.map((line) =>
    line.replace(/^\[[^\]]*\]\s*/, "").trim()
  );
}

module.exports = {
  playSound,
  transcribeWithWhisper
};