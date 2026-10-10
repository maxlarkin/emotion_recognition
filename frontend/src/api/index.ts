// The single data seam for screens. Each group is served by the real backend or by the in-memory
// mock; a later story flips its key to false once the backend serves those endpoints.
import * as mock from "./mock";
import * as real from "./real";
import { connectStreamsSocket } from "./streamsSocket";

export * from "./types";
export { deleteSession, exportUrl, fetchHealth, fetchSessions } from "./real";

const MOCKED = {
  streams: false, // /api/streams*, /ws/streams — real since story 1.7
  system: false, // /api/system — real since story 1.7
  events: true, // /api/events*, incident ack — story 1.10
  settings: true, // /api/settings — story 1.10
  streamLog: true, // /api/streams/{id}/log — story 1.10
  files: true, // /api/files, file sessions, timeline, media — story 1.11
} as const;

export type ApiGroup = keyof typeof MOCKED;
export const isMocked = (group: ApiGroup): boolean => MOCKED[group];

const pick = <T>(group: ApiGroup, m: T, r: T): T => (MOCKED[group] ? m : r);

export const api = {
  listStreams: pick("streams", mock.listStreams, real.listStreams),
  createStream: pick("streams", mock.createStream, real.createStream),
  updateStream: pick("streams", mock.updateStream, real.updateStream),
  deleteStream: pick("streams", mock.deleteStream, real.deleteStream),
  reconnectStream: pick("streams", mock.reconnectStream, real.reconnectStream),
  testStream: pick("streams", mock.testStream, real.testStream),
  connectStreams: pick("streams", mock.connectStreams, connectStreamsSocket),
  fetchSystem: pick("system", mock.fetchSystem, real.fetchSystem),
  fetchStreamLog: pick("streamLog", mock.fetchStreamLog, real.fetchStreamLog),
  listEvents: pick("events", mock.listEvents, real.listEvents),
  ackEvent: pick("events", mock.ackEvent, real.ackEvent),
  exportEvents: pick("events", mock.exportEvents, real.exportEvents),
  fetchSettings: pick("settings", mock.fetchSettings, real.fetchSettings),
  saveSettings: pick("settings", mock.saveSettings, real.saveSettings),
  listFileSessions: pick("files", mock.listFileSessions, real.listFileSessions),
  fetchFileSession: pick("files", mock.fetchFileSession, real.fetchFileSession),
  uploadFile: pick("files", mock.uploadFile, real.uploadFile),
  fetchTimeline: pick("files", mock.fetchTimeline, real.fetchTimeline),
  fileMediaUrl: pick("files", mock.mediaUrl, real.mediaUrl) as (id: string) => string | null,
  fileExportUrl: pick("files", mock.exportUrl, real.exportUrl),
};
