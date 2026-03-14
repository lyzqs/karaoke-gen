import { test, expect } from '@playwright/test';
import { setupApiFixtures, setAuthToken, clearAuthToken } from '../fixtures/test-helper';

/**
 * Regression Tests - Authentication
 *
 * Tests authentication flows using mocked API responses.
 * These tests run offline in CI without hitting production.
 */

test.describe('Authentication - Magic Link', () => {
  test.beforeEach(async ({ page }) => {
    await clearAuthToken(page);
  });

  test('magic link dialog opens from sign in button', async ({ page }) => {
    await setupApiFixtures(page, { mocks: [] });

    await page.goto('/');
    await expect(
      page.getByRole('heading', { name: /create a karaoke video for any song/i })
    ).toBeVisible();

    await page.getByRole('button', { name: /sign in/i }).click();

    const dialog = page.locator('[role="dialog"]');
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText('Enter your email to receive a sign-in link')).toBeVisible();
  });

  test('magic link dialog validates email format', async ({ page }) => {
    await setupApiFixtures(page, { mocks: [] });

    await page.goto('/');
    await expect(
      page.getByRole('heading', { name: /create a karaoke video for any song/i })
    ).toBeVisible();

    await page.getByRole('button', { name: /sign in/i }).click();

    const dialog = page.locator('[role="dialog"]');
    const emailInput = dialog.locator('input[type="email"]');

    // Fill invalid email
    await emailInput.fill('invalid');

    // Try to submit
    await dialog.getByRole('button', { name: /send/i }).click();

    // Should not close dialog (validation failed)
    await expect(dialog).toBeVisible();
  });

  test('magic link request succeeds with valid email', async ({ page }) => {
    await setupApiFixtures(page, {
      mocks: [
        {
          method: 'POST',
          path: '/api/users/auth/magic-link',
          response: {
            status: 200,
            body: { success: true, message: 'Magic link sent' },
          },
        },
      ],
    });

    await page.goto('/');
    await expect(
      page.getByRole('heading', { name: /create a karaoke video for any song/i })
    ).toBeVisible();

    await page.getByRole('button', { name: /sign in/i }).click();

    const dialog = page.locator('[role="dialog"]');
    await dialog.locator('input[type="email"]').fill('test@example.com');
    await dialog.getByRole('button', { name: /send/i }).click();

    // Should show success message
    await expect(page.getByText(/check your email|magic link sent/i)).toBeVisible({
      timeout: 10000,
    });
  });
});

test.describe('Authentication - Token Persistence', () => {
  test('localStorage can store and retrieve tokens', async ({ page }) => {
    // This tests the browser's localStorage functionality
    await page.goto('/');
    await expect(
      page.getByRole('heading', { name: /create a karaoke video for any song/i })
    ).toBeVisible();

    // Set a token
    await page.evaluate(() => {
      localStorage.setItem('test_token', 'my-test-value');
    });

    // Retrieve the token
    const token = await page.evaluate(() => localStorage.getItem('test_token'));
    expect(token).toBe('my-test-value');

    // Token survives navigation
    await page.goto('/#pricing');
    await expect(page.getByRole('heading', { name: /simple pricing/i })).toBeVisible();

    const tokenAfter = await page.evaluate(() => localStorage.getItem('test_token'));
    expect(tokenAfter).toBe('my-test-value');
  });
});
