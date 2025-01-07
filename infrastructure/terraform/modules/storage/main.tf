# Configure AWS provider
provider "aws" {
  # Provider version specified in variables.tf
}

# Data source for availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# KMS key for storage encryption
resource "aws_kms_key" "storage_key" {
  description             = "KMS key for GameGen-X storage encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Environment = var.environment
    Purpose     = "StorageEncryption"
  }
}

# KMS key for backup encryption
resource "aws_kms_key" "backup_key" {
  count                   = var.backup_enabled ? 1 : 0
  description             = "KMS key for GameGen-X backup encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Environment = var.environment
    Purpose     = "BackupEncryption"
  }
}

# MinIO-compatible S3 bucket for video data (100TB)
resource "aws_s3_bucket" "video_storage" {
  bucket_prefix = "gamegen-x-video-${var.environment}"
  force_destroy = false

  tags = {
    Name              = "gamegen-x-video-storage-${var.environment}"
    Environment       = var.environment
    StorageType      = "VideoData"
    FreeBSDCompatible = var.freebsd_compatibility
  }
}

# Video storage bucket versioning
resource "aws_s3_bucket_versioning" "video_versioning" {
  bucket = aws_s3_bucket.video_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Video storage encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "video_encryption" {
  bucket = aws_s3_bucket.video_storage.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.storage_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

# Video storage lifecycle rules
resource "aws_s3_bucket_lifecycle_configuration" "video_lifecycle" {
  bucket = aws_s3_bucket.video_storage.id

  rule {
    id     = "archive_old_data"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = 90
    }
  }
}

# FreeBSD-compatible EBS volume for model weights (10TB)
resource "aws_ebs_volume" "model_storage" {
  availability_zone = data.aws_availability_zones.available.names[0]
  size             = 10240  # 10TB in GB
  type             = "io2"
  iops             = var.iops
  encrypted        = var.encryption_enabled
  kms_key_id       = aws_kms_key.storage_key.arn

  tags = {
    Name              = "gamegen-x-model-storage-${var.environment}"
    Environment       = var.environment
    StorageType      = "ModelWeights"
    FileSystem       = "ZFS"
    FreeBSDCompatible = var.freebsd_compatibility
  }
}

# High-performance EFS for training data (50TB)
resource "aws_efs_file_system" "training_data" {
  creation_token = "gamegen-x-training-data-${var.environment}"
  encrypted      = var.encryption_enabled
  kms_key_id     = aws_kms_key.storage_key.arn

  performance_mode                = "maxIO"
  throughput_mode                = "provisioned"
  provisioned_throughput_in_mibps = var.throughput

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Name              = "gamegen-x-training-data-${var.environment}"
    Environment       = var.environment
    StorageType      = "TrainingData"
    Performance      = "NVMe-optimized"
    FreeBSDCompatible = var.freebsd_compatibility
  }
}

# Mount target for EFS
resource "aws_efs_mount_target" "training_data_mount" {
  file_system_id  = aws_efs_file_system.training_data.id
  subnet_id       = data.aws_subnet.selected.id
  security_groups = [aws_security_group.efs_sg.id]
}

# Security group for EFS
resource "aws_security_group" "efs_sg" {
  name_prefix = "gamegen-x-efs-${var.environment}"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.selected.cidr_block]
  }
}

# Backup vault for storage resources
resource "aws_backup_vault" "storage_backup" {
  count         = var.backup_enabled ? 1 : 0
  name          = "gamegen-x-storage-backup-${var.environment}"
  kms_key_arn   = aws_kms_key.backup_key[0].arn
  force_destroy = false

  tags = {
    Environment       = var.environment
    FreeBSDCompatible = var.freebsd_compatibility
  }
}

# Backup plan for storage resources
resource "aws_backup_plan" "storage_backup" {
  count = var.backup_enabled ? 1 : 0
  name  = "gamegen-x-storage-backup-plan-${var.environment}"

  rule {
    rule_name         = "daily_backup"
    target_vault_name = aws_backup_vault.storage_backup[0].name
    schedule          = "cron(0 5 ? * * *)"
    
    lifecycle {
      cold_storage_after = 30
      delete_after       = 90
    }
  }

  rule {
    rule_name         = "weekly_backup"
    target_vault_name = aws_backup_vault.storage_backup[0].name
    schedule          = "cron(0 5 ? * 1 *)"
    
    lifecycle {
      cold_storage_after = 90
      delete_after       = 365
    }
  }
}

# CloudWatch monitoring for storage resources
resource "aws_cloudwatch_metric_alarm" "storage_usage" {
  count               = var.monitoring_enabled ? 1 : 0
  alarm_name          = "gamegen-x-storage-usage-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "StorageBytes"
  namespace           = "AWS/EFS"
  period             = "300"
  statistic          = "Average"
  threshold          = "85"
  alarm_description  = "Storage usage above 85%"
  alarm_actions      = []

  dimensions = {
    FileSystemId = aws_efs_file_system.training_data.id
  }
}