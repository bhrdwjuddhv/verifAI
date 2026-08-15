import fs from 'fs';
import path from 'path';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';

export type UserRole = 'owner' | 'trainer' | 'viewer';

export interface User {
  id: string;
  email: string;
  passwordHash: string;
  role: UserRole;
  createdAt: string;
  updatedAt: string;
}

/** Permissions map per role */
export const ROLE_PERMISSIONS: Record<UserRole, string[]> = {
  owner: [
    'manage_accounts',
    'promote_checkpoint',
    'run_training',
    'view_runs',
    'scan_admin',
    'view_dashboard',
    'view_audit_log',
  ],
  trainer: [
    'run_training',
    'view_runs',
    'scan_admin',
    'view_dashboard',
  ],
  viewer: [
    'view_dashboard',
    'view_runs',
  ],
};

const BCRYPT_ROUNDS = 12;
const USERS_FILE = path.resolve('data', 'users.json');

function ensureDataDir(): void {
  const dir = path.dirname(USERS_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function readUsers(): User[] {
  ensureDataDir();
  if (!fs.existsSync(USERS_FILE)) {
    return [];
  }
  try {
    const raw = fs.readFileSync(USERS_FILE, 'utf-8');
    return JSON.parse(raw) as User[];
  } catch {
    return [];
  }
}

function writeUsers(users: User[]): void {
  ensureDataDir();
  fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2), 'utf-8');
}

export function getUserByEmail(email: string): User | undefined {
  const users = readUsers();
  return users.find((u) => u.email.toLowerCase() === email.toLowerCase());
}

export function getUserById(id: string): User | undefined {
  const users = readUsers();
  return users.find((u) => u.id === id);
}

export function listUsers(): Omit<User, 'passwordHash'>[] {
  const users = readUsers();
  return users.map(({ passwordHash, ...rest }) => rest);
}

export async function createUser(
  email: string,
  password: string,
  role: UserRole
): Promise<Omit<User, 'passwordHash'>> {
  const users = readUsers();

  // Check duplicate
  if (users.some((u) => u.email.toLowerCase() === email.toLowerCase())) {
    throw new Error(`User with email "${email}" already exists`);
  }

  const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);
  const now = new Date().toISOString();

  const newUser: User = {
    id: crypto.randomUUID(),
    email: email.toLowerCase().trim(),
    passwordHash,
    role,
    createdAt: now,
    updatedAt: now,
  };

  users.push(newUser);
  writeUsers(users);

  const { passwordHash: _, ...safe } = newUser;
  return safe;
}

export async function verifyPassword(user: User, password: string): Promise<boolean> {
  return bcrypt.compare(password, user.passwordHash);
}

export function updateUserRole(id: string, newRole: UserRole): Omit<User, 'passwordHash'> | null {
  const users = readUsers();
  const idx = users.findIndex((u) => u.id === id);
  if (idx === -1) return null;

  users[idx].role = newRole;
  users[idx].updatedAt = new Date().toISOString();
  writeUsers(users);

  const { passwordHash, ...safe } = users[idx];
  return safe;
}

export function deleteUser(id: string): boolean {
  const users = readUsers();
  const filtered = users.filter((u) => u.id !== id);
  if (filtered.length === users.length) return false;

  writeUsers(filtered);
  return true;
}

export function hasPermission(role: UserRole, permission: string): boolean {
  return ROLE_PERMISSIONS[role]?.includes(permission) ?? false;
}

export function getUserCount(): number {
  return readUsers().length;
}
