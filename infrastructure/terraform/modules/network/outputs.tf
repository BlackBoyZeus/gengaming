# VPC identifier output
output "vpc_id" {
  description = "ID of the created VPC for GPU cluster infrastructure"
  value       = aws_vpc.main.id
}

# Public subnet identifier for management access
output "public_subnet_id" {
  description = "ID of the public subnet for management access"
  value       = aws_subnet.public.id
}

# Private subnet identifier for GPU cluster with InfiniBand support
output "private_subnet_id" {
  description = "ID of the private subnet for GPU cluster with InfiniBand support"
  value       = aws_subnet.private.id
}

# Security group identifier for GPU cluster network access
output "gpu_cluster_sg_id" {
  description = "ID of the security group controlling GPU cluster network access"
  value       = aws_security_group.gpu_cluster.id
}

# Network interface identifier for high-speed InfiniBand networking
output "gpu_cluster_eni_id" {
  description = "ID of the network interface supporting 100Gbps InfiniBand for GPU cluster"
  value       = aws_network_interface.gpu_cluster.id
}

# Placement group identifier for optimized GPU cluster networking
output "gpu_cluster_pg_id" {
  description = "ID of the placement group optimizing GPU cluster network latency"
  value       = aws_placement_group.gpu_cluster.id
}

# Comprehensive network configuration output
output "network_config" {
  description = "Combined network configuration for GPU cluster including InfiniBand and security settings"
  value = {
    vpc_id              = aws_vpc.main.id
    subnet_id           = aws_subnet.private.id
    security_group_id   = aws_security_group.gpu_cluster.id
    network_interface_id = aws_network_interface.gpu_cluster.id
    placement_group_id  = aws_placement_group.gpu_cluster.id
    network_performance = {
      bandwidth = "100Gbps"
      latency   = "ultra-low"
      protocol  = "InfiniBand"
    }
    security_zones = {
      public             = aws_subnet.public.id
      private            = aws_subnet.private.id
      management_access  = true
      cluster_isolation  = true
    }
  }
}