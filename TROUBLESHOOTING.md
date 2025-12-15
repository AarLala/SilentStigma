# Troubleshooting Guide

## Memory Issues

### Issue: "Unable to allocate X GiB for an array"

**Problem**: Semantic deduplication tries to create a full similarity matrix which requires too much memory for large datasets.

**Solution**: The code now automatically skips semantic deduplication for datasets larger than 50,000 comments. This is safe because:
- Other deduplication methods (exact duplicates, spam filtering) still work
- Semantic deduplication is computationally expensive and not critical for initial analysis
- You can run it later on smaller subsets if needed

**To disable this safety feature** (not recommended for large datasets):
```yaml
# In config.yaml
processing:
  skip_semantic_dedup_if_large: false
```

### Issue: Out of Memory During Encoding

**Problem**: Encoding 100K+ comments requires significant RAM.

**Solutions**:
1. Process in smaller batches by reducing `max_results_per_channel` in `config.yaml`
2. Close other applications to free up RAM
3. Process fewer channels at a time

## Database Issues

### Issue: "no such table: comments"

**Problem**: Database tables haven't been created yet.

**Solution**: The code now automatically initializes the database when needed. If you still see this error:
1. Delete `data/silencevoice.db` if it exists
2. Run the pipeline again: `python -m src.pipeline --collect`

## API Issues

### Issue: YouTube API Rate Limits

**Problem**: Too many requests cause 403 errors.

**Solutions**:
1. Increase `request_delay` in `config.yaml` (e.g., 0.2 seconds)
2. Reduce `max_results_per_channel` 
3. Wait a few minutes and try again
4. Check your API quota in Google Cloud Console

### Issue: "Comments disabled for video"

**Problem**: Some videos have comments disabled.

**Solution**: This is normal. The pipeline automatically skips these videos and continues.

## Performance Issues

### Issue: Pipeline is very slow

**Solutions**:
1. Reduce dataset size in `config.yaml`
2. Process fewer channels
3. Skip semantic deduplication for large datasets (automatic)
4. Use a machine with more RAM/CPU

## Common Solutions Summary

1. **Memory errors**: Reduce dataset size or skip semantic deduplication
2. **API errors**: Increase delays, reduce requests, check quota
3. **Database errors**: Delete database and re-run collection
4. **Slow processing**: Reduce dataset size, use better hardware

## Getting Help

If issues persist:
1. Check the logs for specific error messages
2. Review `config.yaml` settings
3. Verify your API key is valid
4. Check available disk space and RAM

