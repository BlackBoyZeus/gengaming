# GameGen-X Infrastructure Provider Configuration
# Terraform Version: >= 1.0.0

terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    # AWS Provider for core infrastructure resources
    # Version: ~> 5.0
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    # Null Provider for resource dependencies
    # Version: ~> 3.0
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }

    # Random Provider for unique identifiers
    # Version: ~> 3.0
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }

    # TLS Provider for security configurations
    # Version: ~> 4.0
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

# AWS Provider Configuration
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "GameGen-X"
      Environment = var.environment
      ManagedBy   = "Terraform"
      OS          = "FreeBSD-Orbis"
      Purpose     = "AI-Game-Generation"
      Compliance  = "SOC2"
    }
  }
}

# Null Provider Configuration
# Used for resource dependencies and local operations
provider "null" {
}

# Random Provider Configuration
# Used for generating unique identifiers
provider "random" {
}

# TLS Provider Configuration
# Used for security configurations and certificates
provider "tls" {
}