import { InstanceDetailView } from '@/components/instance-detail-view';

export default function InstanceDetailPage({ params }: { params: { instanceId: string } }) {
  return <InstanceDetailView instanceId={params.instanceId} />;
}
