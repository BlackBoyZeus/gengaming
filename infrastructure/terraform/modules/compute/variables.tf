variable "environment" {
  description = "Environment name for resource tagging (e.g., dev, staging, prod)"
  type        = string
  validation {
    condition     = can(regex("^(dev|staging|prod)$", var.environment))
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "instance_type" {
  description = "GPU instance type based on environment requirements (must support FreeBSD and non-NVIDIA GPU for production)"
  type        = string
  default     = "g4ad.4xlarge"
  validation {
    condition     = can(regex("^g4ad\\.(xlarge|2xlarge|4xlarge|8xlarge|16xlarge)$", var.instance_type))
    error_message = "Instance type must be a valid g4ad series instance supporting AMD GPUs."
  }
}

variable "instance_count" {
  description = "Number of GPU compute instances (min 1 for prod, min 8 for dev/training)"
  type        = number
  default     = 1
  validation {
    condition     = (var.environment == "prod" ? var.instance_count >= 1 : var.instance_count >= 8)
    error_message = "Instance count must be >= 1 for prod and >= 8 for dev/training."
  }
}

variable "root_volume_size" {
  description = "Root volume size in GB (min 500GB for FreeBSD OS and required software)"
  type        = number
  default     = 500
  validation {
    condition     = var.root_volume_size >= 500
    error_message = "Root volume size must be at least 500GB."
  }
}

variable "vpc_id" {
  description = "VPC ID where compute resources will be provisioned"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for instance placement"
  type        = list(string)
}

variable "enable_monitoring" {
  description = "Enable detailed monitoring for instances (recommended for production)"
  type        = bool
  default     = true
}

variable "memory_size" {
  description = "Memory size in GB (min 32GB for prod, min 512GB for dev/training)"
  type        = number
  default     = 32
  validation {
    condition     = (var.environment == "prod" ? var.memory_size >= 32 : var.memory_size >= 512)
    error_message = "Memory size must be >= 32GB for prod and >= 512GB for dev/training."
  }
}

variable "scaling_policy" {
  description = "Auto-scaling policy configuration (CPU-based, GPU-based, or custom)"
  type        = string
  default     = "CPU-based"
  validation {
    condition     = can(regex("^(CPU-based|GPU-based|custom)$", var.scaling_policy))
    error_message = "Scaling policy must be CPU-based, GPU-based, or custom."
  }
}

variable "tags" {
  description = "Additional tags for compute resources"
  type        = map(string)
  default     = {}
}

variable "ami_id" {
  description = "FreeBSD AMI ID optimized for GPU workloads"
  type        = string
}

variable "key_name" {
  description = "SSH key pair name for instance access"
  type        = string
}

variable "security_group_rules" {
  description = "Additional security group rules for compute instances"
  type = list(object({
    type        = string
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
  }))
  default = []
}

variable "ebs_optimized" {
  description = "Enable EBS optimization for better storage performance"
  type        = bool
  default     = true
}

variable "termination_protection" {
  description = "Enable termination protection for production instances"
  type        = bool
  default     = false
}