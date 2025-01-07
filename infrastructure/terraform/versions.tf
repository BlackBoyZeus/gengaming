# GameGen-X Infrastructure Version Constraints
# Terraform Version: >= 1.0.0
# Purpose: Define version constraints for infrastructure deployment on FreeBSD-based Orbis OS

terraform {
  # Require Terraform version 1.0.0 or higher for stability and FreeBSD compatibility
  required_version = ">= 1.0.0"

  required_providers {
    # AWS Provider for core infrastructure resources
    # Used for compute, storage, and networking components
    # Version: ~> 5.0
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    # Null Provider for resource dependencies and orchestration
    # Used for managing deployment sequences and local operations
    # Version: ~> 3.0
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }

    # Random Provider for unique resource identifiers
    # Used for generating unique names and IDs across environments
    # Version: ~> 3.0
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }

    # TLS Provider for security configurations
    # Used for managing certificates and encryption
    # Version: ~> 4.0
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}