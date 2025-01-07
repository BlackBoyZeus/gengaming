# AWS Provider version ~> 5.0 required for network resource validation

variable "environment" {
  type        = string
  description = "Deployment environment name (development, staging, production)"
  validation {
    condition     = can(regex("^(development|staging|production)$", var.environment))
    error_message = "Environment must be development, staging, or production"
  }
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for VPC network segmentation"
  default     = "10.0.0.0/16"
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block"
  }
}

variable "network_speed" {
  type        = string
  description = "Required network speed for GPU cluster (must be specified in Gbps)"
  default     = "100Gbps"
  validation {
    condition     = can(regex("^\\d+Gbps$", var.network_speed))
    error_message = "Network speed must be specified in Gbps (e.g., 100Gbps)"
  }
}

variable "enable_infiniband" {
  type        = bool
  description = "Enable InfiniBand networking for high-speed GPU cluster communication"
  default     = true
}

variable "availability_zones" {
  type        = list(string)
  description = "List of availability zones for subnet distribution and high availability"
  default     = []
  validation {
    condition     = length(var.availability_zones) > 0
    error_message = "At least one availability zone must be specified"
  }
}

variable "security_zones" {
  type = map(object({
    cidr = string
    rules = list(object({
      type        = string
      from_port   = number
      to_port     = number
      protocol    = string
      cidr_blocks = list(string)
    }))
  }))
  description = "Security zone definitions including CIDR blocks and access rules"
  default     = {}
  validation {
    condition = alltrue([
      for zone, config in var.security_zones :
      can(cidrhost(config.cidr, 0)) &&
      alltrue([
        for rule in config.rules :
        contains(["ingress", "egress"], rule.type) &&
        rule.from_port >= 0 && rule.from_port <= 65535 &&
        rule.to_port >= 0 && rule.to_port <= 65535 &&
        contains(["tcp", "udp", "icmp", "-1"], rule.protocol) &&
        alltrue([for cidr in rule.cidr_blocks : can(cidrhost(cidr, 0))])
      ])
    ])
    error_message = "Invalid security zone configuration. Check CIDR blocks, port ranges (0-65535), protocols (tcp, udp, icmp, -1), and rule types (ingress, egress)"
  }
}

variable "tags" {
  type        = map(string)
  description = "Common tags to be applied to all resources"
  default     = {}
}

variable "enable_vpc_flow_logs" {
  type        = bool
  description = "Enable VPC flow logs for network traffic monitoring"
  default     = true
}

variable "vpc_flow_logs_retention" {
  type        = number
  description = "Number of days to retain VPC flow logs"
  default     = 14
  validation {
    condition     = var.vpc_flow_logs_retention >= 1 && var.vpc_flow_logs_retention <= 365
    error_message = "VPC flow logs retention must be between 1 and 365 days"
  }
}

variable "enable_network_acls" {
  type        = bool
  description = "Enable custom network ACLs for additional network security"
  default     = true
}

variable "enable_vpc_endpoints" {
  type        = bool
  description = "Enable VPC endpoints for secure AWS service access"
  default     = true
}