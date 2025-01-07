# GameGen-X Infrastructure Configuration
# Terraform Version: >= 1.0.0

terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket         = "gamegen-x-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = var.region
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
    kms_key_id     = aws_kms_key.terraform_state.id
  }
}

# KMS key for state encryption
resource "aws_kms_key" "terraform_state" {
  description             = "KMS key for Terraform state encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Environment = var.environment
    Project     = "GameGen-X"
    ManagedBy   = "Terraform"
  }
}

# Compute Module - GPU Infrastructure
module "compute" {
  source = "./modules/compute"

  environment         = var.environment
  instance_type      = var.gpu_instance_type
  instance_count     = var.gpu_instance_count
  root_volume_size   = 500
  root_volume_type   = "gp3"
  memory_size        = var.memory_per_node
  enable_infiniBand  = true
  backup_enabled     = true
  monitoring_enabled = true

  tags = {
    Environment = var.environment
    Project     = "GameGen-X"
    Component   = "Compute"
  }
}

# Storage Module - High Performance Storage
module "storage" {
  source = "./modules/storage"

  environment            = var.environment
  storage_size          = var.storage_config.size
  storage_type          = var.storage_config.type
  backup_enabled        = true
  backup_retention      = var.backup_config.retention_days
  encryption_enabled    = true
  performance_mode      = "maxIO"
  throughput_mode       = "provisioned"
  provisioned_throughput = 1024
  multi_az_enabled      = true

  tags = {
    Environment = var.environment
    Project     = "GameGen-X"
    Component   = "Storage"
  }
}

# Network Module - High Performance Networking
module "network" {
  source = "./modules/network"

  environment = var.environment
  vpc_cidr    = "10.0.0.0/16"
  network_speed = var.network_config.speed
  enable_nat   = true
  enable_infiniBand = true
  enable_placement_groups = true

  subnet_configuration = {
    public  = ["10.0.1.0/24", "10.0.2.0/24"]
    private = ["10.0.3.0/24", "10.0.4.0/24"]
  }

  security_groups = {
    compute = {
      ingress_rules = ["ssh", "https", "custom-gpu"]
      egress_rules  = ["all"]
    }
  }

  tags = {
    Environment = var.environment
    Project     = "GameGen-X"
    Component   = "Network"
  }
}

# Monitoring Module - Observability Stack
module "monitoring" {
  source = "./modules/monitoring"

  environment = var.environment
  enable_prometheus = var.monitoring_config.prometheus
  enable_grafana    = true
  enable_jaeger     = var.monitoring_config.jaeger
  enable_elk        = var.monitoring_config.elk
  retention_period  = 30

  alert_configurations = {
    gpu_utilization_threshold = 90
    memory_threshold         = 85
    storage_threshold        = 80
  }

  dashboard_templates = [
    "gpu-metrics",
    "system-metrics",
    "application-metrics"
  ]

  log_aggregation = {
    index_pattern  = "gamegen-x-*"
    retention_days = 30
  }

  tags = {
    Environment = var.environment
    Project     = "GameGen-X"
    Component   = "Monitoring"
  }
}

# Outputs
output "compute_instance_ids" {
  description = "List of provisioned compute instance IDs"
  value       = module.compute.instance_ids
}

output "storage_endpoint" {
  description = "MinIO storage endpoint"
  value       = module.storage.endpoint
}

output "monitoring_endpoints" {
  description = "Monitoring service endpoints"
  value = {
    prometheus_endpoint = module.monitoring.prometheus_endpoint
    grafana_endpoint   = module.monitoring.grafana_endpoint
    kibana_endpoint    = module.monitoring.kibana_endpoint
  }
}

# Common tags for all resources
locals {
  common_tags = {
    Project     = "GameGen-X"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "Platform-Team"
  }
}