import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { resolveRepoPath } from '../persona/repoPaths';
import type { UserModelDoc } from './types';

export function userModelPath(userModelDir: string, userId: string): string {
  return resolveRepoPath(path.join(userModelDir, `${userId}.character.json`));
}

export async function loadUserModelDoc(userModelDir: string, userId: string): Promise<UserModelDoc | null> {
  const p = userModelPath(userModelDir, userId);
  try {
    const raw = await readFile(p, 'utf8');
    return JSON.parse(raw) as UserModelDoc;
  } catch {
    return null;
  }
}

export async function writeUserModelDoc(userModelDir: string, userId: string, doc: UserModelDoc): Promise<void> {
  const p = userModelPath(userModelDir, userId);
  await mkdir(path.dirname(p), { recursive: true });
  await writeFile(p, JSON.stringify(doc, null, 2) + '\n', 'utf8');
}

