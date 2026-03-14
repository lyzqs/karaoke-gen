import { test, expect, type Page } from "@playwright/test"
import { setAuthToken } from "../fixtures/test-helper"

/**
 * Admin Rate Limits Page - Smoke Tests
 *
 * Basic smoke tests for the rate limits admin page.
 * These tests verify the page loads and basic navigation works.
 *
 * More detailed functionality is covered by Jest unit tests.
 */

// Mock data for API responses
const mockBlocklists = {
  disposable_domains: ["tempmail.com", "mailinator.com", "guerrillamail.com"],
  blocked_emails: ["spammer@example.com"],
  blocked_ips: ["192.168.1.100"],
  updated_at: "2025-01-09T10:00:00Z",
  updated_by: "admin@nomadkaraoke.com",
}

const mockUserMe = {
  user: {
    email: "admin@nomadkaraoke.com",
    role: "admin",
    credits: -1,
  },
}

const mockYouTubeQueue = {
  entries: [],
  stats: { queued: 0, processing: 0, failed: 0, completed: 0, total: 0 },
}

async function gotoRateLimitsPage(page: Page) {
  await page.goto("/admin/rate-limits")
  await expect(page).toHaveURL(/\/admin\/rate-limits\/?$/)
  await expect(page.getByRole("heading", { name: /rate limits/i })).toBeVisible()
}

test.describe("Admin Rate Limits Page", () => {
  test.beforeEach(async ({ page }) => {
    // Set auth token for admin access
    await setAuthToken(page, "test-admin-token")

    // Mock shared app-shell requests so page-level smoke tests can settle quickly.
    await page.route("**/api/tenant/config", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          tenant: null,
          is_default: true,
        }),
      })
    })

    await page.route("**/api/info", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          version: "test",
        }),
      })
    })

    await page.route("**/api/health/encoding-worker", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          available: false,
          status: "not_configured",
        }),
      })
    })

    await page.route("**/api/health/flacfetch", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          available: false,
          status: "not_configured",
        }),
      })
    })

    await page.route("**/api/push/vapid-public-key", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          enabled: false,
          vapid_public_key: null,
        }),
      })
    })

    // Mock all API routes that the rate limits page needs
    await page.route("**/api/users/me", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockUserMe),
      })
    })

    await page.route("**/api/admin/rate-limits/blocklists", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockBlocklists),
      })
    })

    await page.route("**/api/admin/rate-limits/youtube-queue", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockYouTubeQueue),
      })
    })

    await page.route("**/api/admin/stats/overview", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          total_users: 100,
          total_jobs: 500,
          active_users: 50,
          jobs_by_status: {},
        }),
      })
    })

    // Mock any other admin API calls with a generic success response
    await page.route("**/api/admin/**", async (route) => {
      if (
        !route.request().url().includes("rate-limits") &&
        !route.request().url().includes("stats")
      ) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ success: true }),
        })
      } else {
        await route.fallback()
      }
    })
  })

  test("rate limits page loads and shows title", async ({ page }) => {
    await gotoRateLimitsPage(page)

    // Check page title
    await expect(page.locator("h1")).toContainText("Rate Limits")
    await expect(
      page.getByText("Manage YouTube upload queue and blocklists")
    ).toBeVisible()
  })

  test("both tabs are visible", async ({ page }) => {
    await gotoRateLimitsPage(page)

    // Check all tabs are present
    await expect(page.getByRole("tab", { name: /youtube queue/i })).toBeVisible()
    await expect(page.getByRole("tab", { name: /blocklists/i })).toBeVisible()
  })

  test("can switch between tabs", async ({ page }) => {
    await gotoRateLimitsPage(page)

    // YouTube Queue is default
    await expect(page.getByRole("tab", { name: /youtube queue/i })).toHaveAttribute(
      "data-state",
      "active"
    )

    // Click blocklists tab
    await page.getByRole("tab", { name: /blocklists/i }).click()
    await expect(page.getByRole("tab", { name: /blocklists/i })).toHaveAttribute(
      "data-state",
      "active"
    )
  })

  test("refresh button is visible and clickable", async ({ page }) => {
    await gotoRateLimitsPage(page)

    // Check refresh button exists and click it
    const refreshButton = page.getByRole("button", { name: /refresh/i })
    await expect(refreshButton).toBeVisible()
    await refreshButton.click()

    // Smoke test: refresh keeps the page interactive and on the same route.
    await expect(refreshButton).toBeVisible()
    await expect(page).toHaveURL(/\/admin\/rate-limits\/?$/)
  })

  test("blocklists tab shows domain search input", async ({ page }) => {
    await gotoRateLimitsPage(page)

    // Navigate to blocklists tab
    await page.getByRole("tab", { name: /blocklists/i }).click()

    // Check search input is visible
    await expect(page.getByPlaceholder("Search domains...")).toBeVisible()
    await expect(page.getByPlaceholder("Search emails...")).toBeVisible()
    await expect(page.getByPlaceholder("Search IPs...")).toBeVisible()
  })
})
