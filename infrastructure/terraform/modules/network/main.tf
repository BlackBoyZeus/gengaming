# AWS Provider version ~> 5.0 required for network resource provisioning
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# VPC for network isolation and GPU cluster deployment
resource "aws_vpc" "main" {
  cidr_block                           = var.vpc_cidr
  enable_dns_hostnames                 = true
  enable_dns_support                   = true
  instance_tenancy                     = "default"
  enable_network_address_usage_metrics = true

  tags = merge(var.tags, {
    Name         = "${var.environment}-vpc"
    Environment  = var.environment
    Purpose      = "GPU-Cluster-Network"
    NetworkSpeed = var.network_speed
  })
}

# Public subnet for management access
resource "aws_subnet" "public" {
  vpc_id                                          = aws_vpc.main.id
  cidr_block                                      = cidrsubnet(var.vpc_cidr, 8, 1)
  availability_zone                               = var.availability_zones[0]
  map_public_ip_on_launch                        = true
  enable_resource_name_dns_a_record_on_launch    = true

  tags = merge(var.tags, {
    Name        = "${var.environment}-public-subnet"
    Environment = var.environment
    NetworkZone = "Public"
    Purpose     = "Management-Access"
  })
}

# Private subnet for GPU cluster compute resources
resource "aws_subnet" "private" {
  vpc_id                                          = aws_vpc.main.id
  cidr_block                                      = cidrsubnet(var.vpc_cidr, 8, 2)
  availability_zone                               = var.availability_zones[0]
  enable_resource_name_dns_a_record_on_launch    = true

  tags = merge(var.tags, {
    Name        = "${var.environment}-private-subnet"
    Environment = var.environment
    NetworkZone = "Private"
    Purpose     = "GPU-Cluster-Compute"
  })
}

# Security group for GPU cluster with InfiniBand networking
resource "aws_security_group" "gpu_cluster" {
  name        = "${var.environment}-gpu-cluster-sg"
  description = "Security group for GPU cluster with high-speed InfiniBand networking"
  vpc_id      = aws_vpc.main.id

  # Allow all TCP traffic within cluster
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
    description = "Allow all TCP traffic within cluster"
  }

  # Allow all UDP traffic within cluster
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "udp"
    self        = true
    description = "Allow all UDP traffic within cluster"
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = merge(var.tags, {
    Name        = "${var.environment}-gpu-cluster-sg"
    Environment = var.environment
    Purpose     = "GPU-Cluster-Security"
  })
}

# Placement group for optimal GPU cluster networking
resource "aws_placement_group" "gpu_cluster" {
  name            = "${var.environment}-gpu-cluster-pg"
  strategy        = "cluster"
  spread_level    = "rack"
  partition_count = 7

  tags = merge(var.tags, {
    Name        = "${var.environment}-gpu-cluster-pg"
    Environment = var.environment
    Purpose     = "GPU-Cluster-Placement"
  })
}

# Network interface with EFA support for high-speed cluster communication
resource "aws_network_interface" "gpu_cluster" {
  subnet_id           = aws_subnet.private.id
  security_groups     = [aws_security_group.gpu_cluster.id]
  interface_type      = var.enable_infiniband ? "efa" : "interface"
  private_ip_list_enabled = true
  ipv4_prefix_count  = 1

  tags = merge(var.tags, {
    Name         = "${var.environment}-gpu-cluster-eni"
    Environment  = var.environment
    NetworkSpeed = var.network_speed
    Purpose      = "GPU-Cluster-Network-Interface"
  })
}

# Internet Gateway for public subnet access
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(var.tags, {
    Name        = "${var.environment}-igw"
    Environment = var.environment
    Purpose     = "Public-Network-Access"
  })
}

# NAT Gateway for private subnet internet access
resource "aws_nat_gateway" "main" {
  subnet_id     = aws_subnet.public.id
  allocation_id = aws_eip.nat.id

  tags = merge(var.tags, {
    Name        = "${var.environment}-nat"
    Environment = var.environment
    Purpose     = "Private-Network-Access"
  })
}

# Elastic IP for NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = merge(var.tags, {
    Name        = "${var.environment}-nat-eip"
    Environment = var.environment
    Purpose     = "NAT-Gateway-IP"
  })
}

# VPC Flow Logs for network monitoring
resource "aws_flow_log" "main" {
  count                = var.enable_vpc_flow_logs ? 1 : 0
  log_destination_type = "cloud-watch-logs"
  log_destination     = aws_cloudwatch_log_group.flow_logs[0].arn
  traffic_type        = "ALL"
  vpc_id              = aws_vpc.main.id

  tags = merge(var.tags, {
    Name        = "${var.environment}-flow-logs"
    Environment = var.environment
    Purpose     = "Network-Monitoring"
  })
}

# CloudWatch Log Group for VPC Flow Logs
resource "aws_cloudwatch_log_group" "flow_logs" {
  count             = var.enable_vpc_flow_logs ? 1 : 0
  name              = "/aws/vpc/flow-logs/${var.environment}"
  retention_in_days = var.vpc_flow_logs_retention

  tags = merge(var.tags, {
    Name        = "${var.environment}-flow-logs"
    Environment = var.environment
    Purpose     = "Network-Monitoring"
  })
}

# Output values for use in other modules
output "vpc_id" {
  value       = aws_vpc.main.id
  description = "ID of the created VPC"
}

output "vpc_arn" {
  value       = aws_vpc.main.arn
  description = "ARN of the created VPC"
}

output "public_subnet_id" {
  value       = aws_subnet.public.id
  description = "ID of the public subnet"
}

output "private_subnet_id" {
  value       = aws_subnet.private.id
  description = "ID of the private subnet"
}

output "security_group_id" {
  value       = aws_security_group.gpu_cluster.id
  description = "ID of the GPU cluster security group"
}

output "placement_group_id" {
  value       = aws_placement_group.gpu_cluster.id
  description = "ID of the GPU cluster placement group"
}