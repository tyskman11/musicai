// Pool-Anbindung fuer MusicAI: verteilte Transkription (Whisper im Browser, jeder Abschnitt wird doppelt geprueft),
// z. B. fuer Liedtexte. Eingang: pool/input (nur dort abgelegte Dateien werden verteilt), Ergebnis: pool/results/<datei>.transcript.json
const path = require('path');
const { createQueue } = require('../../server/chunk-queue');
const log = require('../../server/logger').for('pool-musicai');
const queue = createQueue({ id: 'musicai', dir: __dirname, model: 'onnx-community/whisper-base', language: null, sources: [path.join(__dirname, 'input')], log });

module.exports = {
  id: 'musicai',
  name: 'MusicAI',
  description: 'Lernt den Stil von Musik-Creators (mit Rechte-Bestätigung) und erzeugt neue Musik. Im Pool: Sprach-Transkription von Audio-Abschnitten, z. B. Liedtexte.',
  enabled: true,
  alive: 'http://127.0.0.1:8790/',   // Antwort = start.py laeuft; der Pool startet das Projekt nie selbst
  browserOnly: true,
  createTask: ctx => queue.nextTask(ctx.userId),
  hasWork: () => queue.hasWork(),
  queueSummary: () => queue.summary(),
  queueOverview: () => queue.overview(),
  getChunk: task => queue.getChunk(task),
  acceptResult: (task, result, ctx) => queue.accept(task, result, ctx.userId, ctx.trusted),
};
