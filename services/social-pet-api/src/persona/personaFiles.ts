import { readFile } from 'node:fs/promises';

import type { GeneratedPersonaDoc, PersonaIndexFile } from './types';

export async function loadPersonaDocFromPath(path: string): Promise<GeneratedPersonaDoc> {
  const raw = await readFile(path, 'utf8');
  return JSON.parse(raw) as GeneratedPersonaDoc;
}

export async function loadPersonaIndexFromPath(path: string): Promise<PersonaIndexFile> {
  const raw = await readFile(path, 'utf8');
  return JSON.parse(raw) as PersonaIndexFile;
}

