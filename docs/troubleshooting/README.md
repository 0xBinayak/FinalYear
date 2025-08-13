# Troubleshooting Guide

This directory contains comprehensive troubleshooting guides for the Advanced Federated Pipeline system.

## Quick Reference

### Common Issues
- [System Won't Start](common-issues.md#system-wont-start)
- [Client Connection Problems](common-issues.md#client-connection-problems)
- [Training Failures](common-issues.md#training-failures)
- [Performance Issues](common-issues.md#performance-issues)

### Component-Specific Guides
- [Aggregation Server Issues](aggregation-server-troubleshooting.md)
- [Edge Coordinator Issues](edge-coordinator-troubleshooting.md)
- [SDR Client Issues](sdr-client-troubleshooting.md)
- [Mobile Client Issues](mobile-client-troubleshooting.md)
- [Database Issues](database-troubleshooting.md)

### Operational Guides
- [System Monitoring](../operations/monitoring.md)
- [Log Analysis](../operations/log-analysis.md)
- [Performance Tuning](../operations/performance-tuning.md)
- [Disaster Recovery](../operations/disaster-recovery.md)

## Emergency Procedures

### System Down
1. Check system status: `curl http://localhost:8000/health`
2. Review container status: `docker-compose ps`
3. Check logs: `docker-compose logs -f`
4. Restart services: `docker-compose restart`

### Data Loss
1. Stop all services immediately
2. Assess damage scope
3. Restore from latest backup
4. Verify data integrity
5. Resume operations

### Security Incident
1. Isolate affected components
2. Preserve logs and evidence
3. Notify security team
4. Follow incident response plan

## Diagnostic Tools

### Health Checks
```bash
# System health
curl http://localhost:8000/health

# Component health
curl http://localhost:8001/health  # Edge Coordinator
curl http://localhost:9090/health  # Monitoring

# Database health
docker-compose exec postgres pg_isready
```

### Log Collection
```bash
# Collect all logs
docker-compose logs > system-logs-$(date +%Y%m%d-%H%M%S).log

# Component-specific logs
docker-compose logs aggregation-server
docker-compose logs edge-coordinator
docker-compose logs postgres
```

### Performance Monitoring
```bash
# System resources
docker stats
htop

# Network connectivity
ping aggregation-server
telnet localhost 8000

# Database performance
docker-compose exec postgres psql -c "SELECT * FROM pg_stat_activity;"
```

## Support Contacts

- **System Administrator**: admin@company.com
- **Development Team**: dev-team@company.com
- **Security Team**: security@company.com
- **24/7 Support**: +1-555-SUPPORT

## Escalation Matrix

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| Critical | 15 minutes | Immediate |
| High | 1 hour | 2 hours |
| Medium | 4 hours | 8 hours |
| Low | 24 hours | 48 hours |