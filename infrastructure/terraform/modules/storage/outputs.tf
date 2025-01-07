# Output for video storage bucket ID (MinIO-compatible S3)
output "video_storage_bucket" {
  description = "ID of the S3 bucket used for video data storage (100TB capacity)"
  value       = aws_s3_bucket.video_storage.id
}

# Output for video storage bucket ARN
output "video_storage_arn" {
  description = "ARN of the S3 bucket used for video data storage, compatible with MinIO"
  value       = aws_s3_bucket.video_storage.arn
}

# Output for model storage volume ID (ZFS)
output "model_storage_id" {
  description = "ID of the EBS volume used for model weights storage (10TB ZFS)"
  value       = aws_ebs_volume.model_storage.id
}

# Output for model storage volume ARN
output "model_storage_arn" {
  description = "ARN of the EBS volume used for model weights storage"
  value       = aws_ebs_volume.model_storage.arn
}

# Output for training data filesystem ID (NVMe)
output "training_data_id" {
  description = "ID of the EFS filesystem used for training data storage (50TB NVMe)"
  value       = aws_efs_file_system.training_data.id
}

# Output for training data filesystem ARN
output "training_data_arn" {
  description = "ARN of the EFS filesystem used for training data storage"
  value       = aws_efs_file_system.training_data.arn
}

# Output for training data mount target ID
output "training_data_mount_target_id" {
  description = "ID of the EFS mount target for training data access"
  value       = aws_efs_mount_target.training_data_mount.id
}

# Output for storage encryption key ARN
output "storage_kms_key_arn" {
  description = "ARN of the KMS key used for storage encryption"
  value       = aws_kms_key.storage_key.arn
}

# Output for backup vault ARN (conditional)
output "backup_vault_arn" {
  description = "ARN of the backup vault when backups are enabled for storage resources"
  value       = var.backup_enabled ? aws_backup_vault.storage_backup[0].arn : null
}

# Output for backup encryption key ARN (conditional)
output "backup_kms_key_arn" {
  description = "ARN of the KMS key used for backup encryption when backups are enabled"
  value       = var.backup_enabled ? aws_kms_key.backup_key[0].arn : null
}

# Output for EFS security group ID
output "efs_security_group_id" {
  description = "ID of the security group controlling EFS access"
  value       = aws_security_group.efs_sg.id
}

# Output for storage monitoring alarm ARN (conditional)
output "storage_alarm_arn" {
  description = "ARN of the CloudWatch alarm monitoring storage usage when monitoring is enabled"
  value       = var.monitoring_enabled ? aws_cloudwatch_metric_alarm.storage_usage[0].arn : null
}