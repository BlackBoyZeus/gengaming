# Provider version constraints and requirements
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0.0"
}

# Data source for FreeBSD AMI
data "aws_ami" "freebsd" {
  most_recent = true
  owners      = ["679593333241"] # Official FreeBSD AMI owner ID

  filter {
    name   = "name"
    values = ["FreeBSD 13.2-RELEASE-amd64*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# GPU-enabled compute instances for GameGen-X
resource "aws_instance" "compute" {
  count = var.instance_count

  ami           = coalesce(var.ami_id, data.aws_ami.freebsd.id)
  instance_type = var.instance_type

  subnet_id                   = element(var.subnet_ids, count.index)
  vpc_security_group_ids      = [aws_security_group.compute.id]
  key_name                   = var.key_name
  disable_api_termination    = var.environment == "prod" ? var.termination_protection : false
  ebs_optimized             = var.ebs_optimized
  monitoring                = var.enable_monitoring

  # Root volume configuration based on environment
  root_block_device {
    volume_size = var.environment == "prod" ? var.root_volume_size : 50000 # 50TB for dev/training
    volume_type = "gp3"
    iops        = 16000
    throughput  = 1000
    encrypted   = true
    tags = {
      Name        = "${var.environment}-gpu-compute-${count.index + 1}-root"
      Environment = var.environment
    }
  }

  # Enhanced networking for better performance
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  # User data script for FreeBSD configuration
  user_data = templatefile("${path.module}/scripts/setup_freebsd.sh", {
    environment  = var.environment
    memory_size  = var.memory_size
    gpu_type     = "AMD"
  })

  # Resource tags
  tags = merge(
    var.tags,
    {
      Name         = "${var.environment}-gpu-compute-${count.index + 1}"
      Environment  = var.environment
      Purpose      = "GameGen-X-Compute"
      GPUType      = "AMD"
      OSType       = "FreeBSD"
      MemorySize   = var.memory_size
      StorageType  = "NVMe"
    }
  )

  lifecycle {
    create_before_destroy = true
    ignore_changes       = [ami] # Allow AMI updates through other processes
  }
}

# Security group for compute instances
resource "aws_security_group" "compute" {
  name_prefix = "${var.environment}-compute-sg"
  vpc_id      = var.vpc_id
  description = "Security group for GameGen-X compute instances"

  # SSH access from internal network
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
    description = "SSH access from internal network"
  }

  # FastAPI service port
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
    description = "FastAPI service port"
  }

  # GPU metrics monitoring port
  ingress {
    from_port   = 9400
    to_port     = 9400
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
    description = "GPU metrics monitoring"
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  # Dynamic security group rules
  dynamic "ingress" {
    for_each = var.security_group_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
      description = "Custom rule ${ingress.key + 1}"
    }
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.environment}-compute-sg"
      Environment = var.environment
      Purpose     = "GameGen-X-Security"
    }
  )
}

# CloudWatch alarms for GPU monitoring
resource "aws_cloudwatch_metric_alarm" "gpu_utilization" {
  count               = var.environment == "prod" ? var.instance_count : 0
  alarm_name          = "${var.environment}-gpu-utilization-${count.index + 1}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "GPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "90"
  alarm_description   = "GPU utilization above 90%"
  alarm_actions      = []  # Add SNS topic ARN for notifications

  dimensions = {
    InstanceId = aws_instance.compute[count.index].id
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.environment}-gpu-alarm-${count.index + 1}"
      Environment = var.environment
    }
  )
}

# Output values for use in other modules
output "instance_ids" {
  description = "IDs of created compute instances"
  value       = aws_instance.compute[*].id
}

output "private_ips" {
  description = "Private IP addresses of compute instances"
  value       = aws_instance.compute[*].private_ip
}

output "security_group_id" {
  description = "ID of the compute security group"
  value       = aws_security_group.compute.id
}

output "instance_arns" {
  description = "ARNs of compute instances"
  value       = aws_instance.compute[*].arn
}