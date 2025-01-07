# List of provisioned GPU compute instance IDs
output "instance_ids" {
  description = "List of IDs of provisioned GPU compute instances for H800 nodes with 80GB VRAM"
  value       = aws_instance.compute[*].id
}

# List of private IP addresses for internal communication
output "private_ips" {
  description = "List of private IP addresses for internal communication between compute nodes over 100Gbps InfiniBand"
  value       = aws_instance.compute[*].private_ip
}

# List of public IP addresses for external access
output "public_ips" {
  description = "List of public IP addresses for external monitoring and management access"
  value       = aws_instance.compute[*].public_ip
}

# Security group ID for network access control
output "security_group_id" {
  description = "ID of the security group controlling network access for FreeBSD-based compute instances"
  value       = aws_security_group.compute.id
}