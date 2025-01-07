# Provider version constraints
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# Environment variable - controls deployment target
variable "environment" {
  type        = string
  description = "Deployment environment (development, staging, production)"
  
  validation {
    condition     = can(regex("^(development|staging|production)$", var.environment))
    error_message = "Environment must be development, staging, or production"
  }
}

# Storage type variable - determines volume type
variable "storage_type" {
  type        = string
  description = "Type of storage volume (gp3-nvme, io2, zfs)"
  
  validation {
    condition     = can(regex("^(gp3-nvme|io2|zfs)$", var.storage_type))
    error_message = "Storage type must be gp3-nvme, io2, or zfs"
  }
}

# Storage size variable - configures volume capacity
variable "storage_size" {
  type        = number
  description = "Storage size in TB (10-100TB based on type)"
  
  validation {
    condition     = var.storage_size >= 10 && var.storage_size <= 100
    error_message = "Storage size must be between 10TB and 100TB"
  }
}

# Backup configuration variable
variable "backup_enabled" {
  type        = bool
  description = "Enable AWS Backup for storage resources with daily snapshots"
  default     = true
}

# Encryption configuration variable
variable "encryption_enabled" {
  type        = bool
  description = "Enable AES-256 storage encryption"
  default     = true
}

# Storage throughput variable
variable "throughput" {
  type        = number
  description = "Provisioned throughput in MiB/s (minimum 1000 for training data)"
  default     = 1000
  
  validation {
    condition     = var.throughput >= 1000
    error_message = "Throughput must be at least 1000 MiB/s for training data requirements"
  }
}

# IOPS configuration variable
variable "iops" {
  type        = number
  description = "Provisioned IOPS for io2 volumes"
  default     = 50000
  
  validation {
    condition     = var.iops >= 10000
    error_message = "IOPS must be at least 10000 for performance requirements"
  }
}

# ZFS snapshot configuration variable
variable "snapshot_frequency" {
  type        = string
  description = "Frequency of ZFS snapshots (hourly, daily, weekly)"
  default     = "daily"
  
  validation {
    condition     = can(regex("^(hourly|daily|weekly)$", var.snapshot_frequency))
    error_message = "Snapshot frequency must be hourly, daily, or weekly"
  }
}

# FreeBSD compatibility variable
variable "freebsd_compatibility" {
  type        = bool
  description = "Enable FreeBSD-specific storage optimizations"
  default     = true
}

# Monitoring configuration variable
variable "monitoring_enabled" {
  type        = bool
  description = "Enable CloudWatch detailed monitoring for storage metrics"
  default     = true
}