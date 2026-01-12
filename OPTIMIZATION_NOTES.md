# Performance Optimizations & Spam Protection

## Gunicorn Compatibility ✅

- App properly exports `application` variable for Gunicorn
- Uses `--preload` flag for faster startup
- 3 workers with 4 threads each for optimal performance
- Proper worker class (`sync`) for Flask

## Database Loading Optimizations ✅

### Supabase Queries
- **Count queries**: Use `count='exact'` with `limit(0)` for fastest counts
- **Caching**: All stats cached in memory (`PRESTORED_METRICS`)
- **Pre-loading**: Common searches pre-loaded at startup
- **Connection pooling**: Supabase client handles pooling automatically

### CSV Data Loading
- **Pre-computed columns**: `text_lower` column created once
- **Optimized dtypes**: Specified dtypes for faster CSV reading
- **In-memory cache**: Data loaded once, reused for all requests
- **Thread-safe**: Proper locking for multi-worker Gunicorn

## Spam & Overload Protection ✅

### Rate Limiting (Per Endpoint)
- **Search**: 30 requests/minute per IP
- **Stats**: 60 requests/minute per IP
- **Clusters**: 30 requests/minute per IP
- **Track Search**: 100 requests/minute per IP
- **Track Download**: 10 requests/minute per IP (strict)
- **Track Session**: 5 requests/minute per IP (very strict)
- **Exports**: 3-5 requests/hour per IP (very strict)

### Additional Protection
- **Query sanitization**: Prevents SQL injection, XSS
- **Query length limits**: Max 200 characters
- **Cache headers**: Reduces redundant requests
- **Security headers**: XSS protection, frame options
- **Request validation**: All inputs sanitized

## Caching Strategy ✅

### In-Memory Cache
- **Search results**: 500 query cache (LRU eviction)
- **Cluster stats**: Cached on first load
- **Metrics**: Pre-stored in memory (updated periodically)

### HTTP Cache Headers
- **Stats**: 10 minutes cache
- **Clusters**: 10 minutes cache
- **Search results**: 5 minutes cache

## Optional: Redis for Better Performance

If you want even better performance with multiple Gunicorn workers:

1. **Add Redis** (Render has Redis addon):
   ```bash
   # In Render dashboard, add Redis service
   # Then set REDIS_URL environment variable
   ```

2. **Benefits**:
   - Shared cache across all workers
   - Better rate limiting (shared across workers)
   - Faster performance

3. **Uncomment in requirements.txt**:
   ```
   redis>=5.0.0
   ```

## Database Recommendations

### Current: Supabase (Best Choice)
- ✅ Fast PostgreSQL queries
- ✅ Built-in connection pooling
- ✅ Automatic scaling
- ✅ Free tier available

### Alternative: Add Redis Cache Layer
- Use Redis to cache frequent queries
- Reduces database load
- Faster response times

## Monitoring

Check `/health` endpoint for:
- Cache status
- Data loading status
- Supabase connection status

