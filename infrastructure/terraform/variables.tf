# Terraform variables definition file for GameGen-X infrastructure
# AWS Provider Version: ~> 5.0

# Environment configuration
variable "environment" {
  type        = string
  description = "Deployment environment (development, staging, production)"
  validation {
    condition     = can(regex("^(development|staging|production)$", var.environment))
    error_message = "Environment must be development, staging, or production"
  }
}

# Region configuration
variable "region" {
  type        = string
  description = "Infrastructure deployment region"
}

# Operating system configuration
variable "os_type" {
  type        = string
  description = "Operating system type for instances"
  validation {
    condition     = can(regex("^(freebsd|ubuntu)$", var.os_type))
    error_message = "OS type must be either freebsd or ubuntu"
  }
}

# GPU instance configuration
variable "gpu_instance_type" {
  type        = string
  description = "GPU instance type for compute nodes"
  default     = "h800.24xlarge"
}

variable "gpu_instance_count" {
  type        = number
  description = "Number of GPU instances per environment"
  validation {
    condition     = var.gpu_instance_count >= 8 && var.gpu_instance_count <= 24
    error_message = "GPU instance count must be between 8 and 24"
  }
}

# Memory configuration
variable "memory_per_node" {
  type        = number
  description = "Memory per node in GB"
  validation {
    condition     = var.memory_per_node >= 32 && var.memory_per_node <= 512
    error_message = "Memory per node must be between 32GB and 512GB"
  }
}

# Storage configuration
variable "storage_config" {
  type = object({
    type = string
    size = number
  })
  description = "Storage configuration"
  validation {
    condition     = contains(["nvme", "ssd"], var.storage_config.type) && var.storage_config.size >= 10 && var.storage_config.size <= 100
    error_message = "Invalid storage configuration"
  }
}

# Network configuration
variable "network_config" {
  type = object({
    speed = string
    type  = string
  })
  description = "Network configuration"
  default = {
    speed = "100Gbps"
    type  = "infiniband"
  }
}

# Monitoring configuration
variable "monitoring_config" {
  type = object({
    prometheus = bool
    elk       = bool
    jaeger    = bool
  })
  description = "Monitoring stack configuration"
  default = {
    prometheus = true
    elk       = true
    jaeger    = true
  }
}

# Backup configuration
variable "backup_config" {
  type = object({
    retention_days = number
    frequency     = string
    type          = string
  })
  description = "Backup and disaster recovery configuration"
  validation {
    condition     = var.backup_config.retention_days >= 7 && contains(["daily", "weekly", "monthly"], var.backup_config.frequency)
    error_message = "Invalid backup configuration"
  }
}