# Metrics Tracking Verification

This document verifies that the implementation meets all the specified criteria.

## ✅ Criteria 1: Searches - No Restriction, Increment Every Time

**Requirement:** Every time the search button is clicked, increment by 1. No restrictions.

**Implementation:**
- ✅ Search tracking happens in `runGlobalSearch()` function in `index.html`
- ✅ Calls `/api/track/search` endpoint every time a search is performed
- ✅ `/api/track/search` endpoint calls `_track_search_supabase()` which increments the count
- ✅ No restrictions - every search increments the counter
- ✅ Search count is displayed in the grey badge: "9,382 searches"
- ✅ Count updates automatically after each search

**Code Locations:**
- Frontend: `src/dashboard/templates/index.html` - `runGlobalSearch()` function
- Backend: `src/dashboard/app.py` - `_track_search_supabase()` and `/api/track/search` endpoint

**Test:** Click the search button multiple times → count should increment each time.

---

## ✅ Criteria 2: Downloads - Once Per Computer

**Requirement:** Every computer can only download once. Keep track of the system. All download buttons must follow this rule.

**Implementation:**
- ✅ Client ID is generated from browser headers (User-Agent + Accept-Language hash)
- ✅ `_track_download_supabase(client_id)` checks if client already downloaded
- ✅ Uses UNIQUE constraint on `client_id` in `download_events` table
- ✅ Only increments download count if it's the first download for that client
- ✅ All download buttons are covered:
  1. ✅ "Download processed dataset" button → calls `/api/export/processed`
  2. ✅ "Download cluster dataset" button → calls `/api/export/all`
  3. ✅ Individual cluster export buttons → calls `/api/export/cluster/<id>`
- ✅ All export endpoints track downloads before serving the file

**Code Locations:**
- Client ID generation: `src/dashboard/app.py` - `get_client_id()` function
- Download tracking: `src/dashboard/app.py` - `_track_download_supabase()` function
- Export endpoints: 
  - `/api/export/processed` (line 438)
  - `/api/export/all` (line 419)
  - `/api/export/cluster/<id>` (line 390)

**Test:** 
1. Download a file from one computer → count increments
2. Try to download again from same computer → count does NOT increment
3. Download from different computer → count increments

---

## ✅ Criteria 3: Exploratory Sessions - Once Per Hour Per Computer

**Requirement:** If a computer does another session, it shouldn't count up unless it has been an hour since the first one.

**Implementation:**
- ✅ `_track_session_supabase(client_id)` checks last session timestamp
- ✅ Queries `session_events` table for the most recent session for that client
- ✅ Only increments if:
  - No previous session exists, OR
  - Last session was more than 1 hour ago
- ✅ Uses `timedelta(hours=1)` to check the time difference
- ✅ Called automatically when dashboard loads via `initClientIdAndImpact()`

**Code Locations:**
- Session tracking: `src/dashboard/app.py` - `_track_session_supabase()` function (line 604)
- Initialization: `src/dashboard/templates/index.html` - `initClientIdAndImpact()` function
- Endpoint: `/api/track/session` (line 731)

**Test:**
1. Load dashboard → session count increments
2. Refresh immediately → session count does NOT increment
3. Wait 1 hour and refresh → session count increments

---

## Summary

| Metric | Requirement | Status | Implementation |
|--------|------------|--------|----------------|
| **Searches** | Increment every time, no restriction | ✅ | Frontend calls `/api/track/search` on every search |
| **Downloads** | Once per computer | ✅ | Backend checks `download_events` table with UNIQUE constraint |
| **Sessions** | Once per hour per computer | ✅ | Backend checks last session timestamp, only increments if >1 hour |

## Database Schema

All metrics are stored in Supabase:

1. **`metrics` table**: Stores the counts
   - `key`: 'searches', 'downloads', 'exploratory_sessions'
   - `value`: integer count

2. **`download_events` table**: Tracks unique downloads
   - `client_id`: unique identifier per computer (UNIQUE constraint)
   - `timestamp`: when download occurred

3. **`session_events` table**: Tracks exploratory sessions
   - `client_id`: identifier per computer
   - `timestamp`: when session occurred
   - Multiple rows per client allowed (one per hour)

## Client ID Generation

Client ID is generated from browser headers to create a stable identifier per computer:
- Uses `User-Agent` + `Accept-Language` headers
- Hashed with MD5 to create consistent ID
- Same computer = same client_id (unless browser/headers change)

## Testing Checklist

- [ ] Search button increments count every time (no restrictions)
- [ ] Download from same computer only counts once
- [ ] Download from different computer counts separately
- [ ] All download buttons (3 locations) track correctly
- [ ] Session increments on first load
- [ ] Session does NOT increment on immediate refresh
- [ ] Session increments after 1 hour wait
- [ ] Search count displays in grey badge
- [ ] Metrics update in real-time on dashboard

