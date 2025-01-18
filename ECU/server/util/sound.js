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
            const player = null
            if (process.env.OS === "ubuntu")
                player = spawn('aplay', [filePath]);
            else
                player = spawn('aplay', ['-D', 'plughw:$(aplay -l | grep -i "usb audio" | grep -io "card [0-9]" | grep -o [0-9]),0', filePath]);

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
 * Records audio using ffmpeg and saves it as a WAV file in 16kHz mono format.
 * @param {string} outputFilePath - Path to save the recorded WAV file.
 * @param {number} durationInSeconds - Duration to record in seconds.
 * @returns {Promise<void>}
 */
function recordAudioWithFFmpeg(outputFilePath, durationInSeconds = 5) {
    return new Promise((resolve, reject) => {

        ffmpeg()
            .input('default')
            .inputFormat('alsa')
            .audioFrequency(16000)
            .audioChannels(1)
            .audioCodec('pcm_s16le')
            .format('wav')
            .duration(durationInSeconds)
            .on('end', () => {
                resolve();
            })
            .on('error', (err) => {
                reject(err);
            })
            .save(outputFilePath);
    });
}

/**
 * Prompts the user for a confirmation of alertness by recording and analyzing audio.
 * @returns {Promise<number>} - User status based on their response.
 */
async function askForUserConfirmation() {
    const userAudioPath = path.join(__dirname, "../assets/sounds", "userCollected.wav");
    // Record audio using ffmpeg
    await recordAudioWithFFmpeg(userAudioPath, 4);
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