/**
 * @file audioProcessing.js
 * @description Handles audio processing tasks such as recording, playing sounds, and transcribing audio files.
 * @author Lior Jigalo
 * @license MIT
 */
const { exec, execFile } = require('child_process');
const path = require("path");
const { spawn } = require('child_process');

const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');

const {confirmationPhrases, noResponsePhrases} = require("./const");
const {user_status, locks} = require("./global");
const {printToConsole} = require("./util");

const natural = require('natural');
const stemmer = natural.PorterStemmer; // Using Porter Stemmer

const tokenizer = new natural.WordTokenizer();

/**
 * Mutes all audio streams using a linux command.
 */
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

/**
 * Unmutes all audio streams using a linux command.
 */
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

/**
 * Plays a sound file.
 * @param {string} filePath - Path to the WAV file to play.
 * @returns {Promise<void>} - Resolves when the sound has finished playing.
 */
function playSound(filePath) {
    return new Promise((resolve, reject) => {
        // muteAllStreams();
            var player = null
            if (process.env.OS === "ubuntu")
                player = spawn('aplay', [filePath]);
            else
                player = spawn('aplay', ['-D', 'plughw:3,0', filePath]);

        player.on('error', (err) => {
            console.error(`Error playing sound: ${err.message}`);
            reject(err);
        });

        player.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Sound player exited with code ${code}`));
                locks.alert_lock = false;
            } else {
                resolve();
            }
        });
        // setTimeout(unmuteAllStreams, 4000);
    });
}

/** Preprocesses confirmation phrases with stemming. */
const stemmedConfirmationPhrases = confirmationPhrases.map(phrase => {
    return phrase.split(' ').map(word => stemmer.stem(word)).join(' ');
});

/** Preprocesses no-response phrases with stemming. */
const stemmedNoResponsePhrases = noResponsePhrases.map(phrase => {
    return phrase.split(' ').map(word => stemmer.stem(word)).join(' ');
});

/**
 * Checks if the transcription includes a confirmation of alertness.
 * @param {Array<string>} lines - Parsed transcription lines.
 * @returns {number} - User status indicating the response type.
 */
function hasAlertConfirmation(lines) {
    for (const line of lines) {
        // Tokenize the line into words
        const words = tokenizer.tokenize(line.toLowerCase());

        // Stem each word
        const stemmedWords = words.map(word => stemmer.stem(word));

        // Reconstruct the line from stemmed words
        const stemmedLine = stemmedWords.join(' ');

        // Check against each stemmed no response phrase
        for (const phrase of stemmedNoResponsePhrases) {
            if (stemmedLine.includes(phrase))
                return user_status.noResponse;
        }

        // Check against each stemmed confirmation phrase
        for (const phrase of stemmedConfirmationPhrases) {
            if (stemmedLine.includes(phrase))
                return user_status.userResponded;
        }
    }

    if (lines.length === 0)
        return user_status.noResponse;
    return user_status.failedToParse;
}

/**
 * Transcribes a WAV file using whisper-cli.
 * @param {string} audioFilePath - The path to the .wav file.
 * @returns {Promise<string>} - The raw transcription output from whisper-cli.
 */
async function transcribeWithWhisper(audioFilePath) {
  // Path to the whisper-cli binary
  const whisperCliPath = path.join(__dirname, "../assets/bin", "whisper-cli");
  // Path to the whisper model
  const modelPath = path.join(__dirname, "../assets/models", "ggml-tiny.en.bin");

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


/**
 * Parses the raw transcription lines from whisper-cli output.
 * @param {string} rawOutput - Raw output from whisper-cli.
 * @returns {Array<string>} - Parsed transcription lines.
 */
function parseWhisperOutput(rawOutput) {

  // Remove bracketed part using a regex and keep text portion
  const lines = rawOutput.split("\n").filter(Boolean);

  return lines.map((line) =>
    line.replace(/^\[[^\]]*\]\s*/, "").trim()
  );
}


/**
 * Validates that the output directory exists and is writable.
 * @param {string} filePath - Path to the output file.
 * @throws Will throw an error if the directory doesn't exist or isn't writable.
 */
function validateOutputDirectory(filePath) {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
        throw new Error(`Output directory does not exist: ${dir}`);
    }
    try {
        fs.accessSync(dir, fs.constants.W_OK);
    } catch (err) {
        throw new Error(`No write permission for directory: ${dir}`);
    }
}

/**
 * Retrieves the ALSA device identifier for USB audio devices.
 * @returns {Promise<string[]>} - Resolves with an array of device identifiers (e.g., ['hw:2,0']) or an empty array if none found.
 */
function getUSBAudioDevices() {
    return new Promise((resolve, reject) => {
        exec('arecord -l', (error, stdout, stderr) => {
            if (error) {
                console.error('Error executing arecord:', stderr);
                return reject(new Error('Failed to list audio capture devices.'));
            }

            const devices = [];
            const lines = stdout.split('\n');

            lines.forEach(line => {
                const usbAudioMatch = line.match(/card\s+(\d+):\s+.*USB.*device\s+(\d+):/i);
                if (usbAudioMatch) {
                    const card = usbAudioMatch[1];
                    const device = usbAudioMatch[2];
                    devices.push(`hw:${card},${device}`);
                }
            });

            resolve(devices);
        });
    });
}

/**
 * Selects the first available USB audio device.
 * @returns {Promise<string>} - Resolves with the device identifier (e.g., 'hw:2,0') or rejects if none found.
 */
async function selectAudioDevice() {
    const devices = await getUSBAudioDevices();
    if (devices.length === 0) {
        throw new Error('No USB audio capture devices found.');
    }
    return devices[0]; // Select the first device
}


/**
 * Records audio using FFmpeg and saves it as a WAV file in 16kHz mono format.
 * Automatically detects and selects the USB audio device.
 * @param {string} outputFilePath - Path to save the recorded WAV file.
 * @param {number} [durationInSeconds=5] - Duration to record in seconds.
 * @param {boolean} [verbose=false] - Enable verbose logging.
 * @returns {Promise<void>}
 */
async function recordAudioWithFFmpeg(outputFilePath, durationInSeconds = 5, verbose = false) {
    try {
        // Validate output directory
        validateOutputDirectory(outputFilePath);

        // Select the audio device
        const audioDevice = await selectAudioDevice();

        if (verbose) {
            console.log(`Using USB Audio Device: ${audioDevice}`);
        }

        return new Promise((resolve, reject) => {
            // Initialize FFmpeg command
            let command = ffmpeg()
                .input(`plug${audioDevice}`)
                .inputFormat('alsa')
                .audioFrequency(16000)         // 16kHz
                .audioChannels(1)              // Mono
                .audioCodec('pcm_s16le')       // PCM 16-bit little endian
                .format('wav')
                .duration(durationInSeconds)
                .save(outputFilePath);

            // Enable verbose logging if requested
            if (verbose) {
                command = command
                    .on('start', (cmdLine) => {
                        console.log(`FFmpeg process started: ${cmdLine}`);
                    })
                    .on('progress', (progress) => {
                        if (progress.percent) {
                            console.log(`Processing: ${progress.percent.toFixed(2)}% done`);
                        }
                    });
            }

            // Handle process end
            command.on('end', () => {
                if (verbose) {
                    console.log('FFmpeg recording finished successfully.');
                }
                resolve();
            });

            // Handle errors
            command.on('error', (err, stdout, stderr) => {
                console.error('FFmpeg error:', err.message);
                if (verbose) {
                    console.error('FFmpeg stdout:', stdout);
                    console.error('FFmpeg stderr:', stderr);
                }
                reject(new Error(`FFmpeg recording failed: ${err.message}`));
            });
        });
    } catch (error) {
        console.error('Error in recordAudioWithFFmpeg:', error.message);
        throw error; // Re-throw the error after logging
    }
}

/**
 * Prompts the user for a confirmation of alertness by recording and analyzing audio.
 * @returns {Promise<number>} - User status based on their response.
 */
async function askForUserConfirmation() {
    const userAudioPath = path.join(__dirname, "../assets/sounds", "userCollected.wav");
    // Record audio using ffmpeg
    await recordAudioWithFFmpeg(userAudioPath, 2);
    // Get output from whisper-cli
    const rawOutput = await transcribeWithWhisper(userAudioPath);
    // Parse timestamps
    const linesWithoutTimestamps = parseWhisperOutput(rawOutput);
    printToConsole(linesWithoutTimestamps)

    // check user response
    return  hasAlertConfirmation(linesWithoutTimestamps);
}

/** Exported functions */
module.exports = {
    playSound,
    askForUserConfirmation
};