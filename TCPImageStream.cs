using UnityEngine;
using System.Threading;
using System.Net.Sockets;
using System.IO;
using System.Collections.Concurrent;
using System.Text;
using System.Net;

public class TCPImageStream : MonoBehaviour
{
    [SerializeField]
    private string ipAddress = "127.0.0.1";
    [SerializeField]
    private int ipPort = 8083;

    Thread m_NetworkThread;
    bool m_NetworkRunning;
    ConcurrentQueue<byte[]> dataQueue = new ConcurrentQueue<byte[]>();
    private void OnEnable()
    {
        m_NetworkRunning = true;
        m_NetworkThread = new Thread(NetworkThread);
        m_NetworkThread.Start();

    }
    private void OnDisable()
    {
        m_NetworkRunning = false;
        if (m_NetworkThread != null)
        {
            if (!m_NetworkThread.Join(100))
            {
                m_NetworkThread.Abort();
            }
        }
    }
    private void NetworkThread()
    {
        TcpClient client = new TcpClient();
        client.Connect(ipAddress, ipPort);
        Debug.Log($"Connecting to {ipAddress}/{ipPort}");
        using (var stream = client.GetStream())
        {
            BinaryReader reader = new BinaryReader(stream);
            try
            {
                while (m_NetworkRunning && client.Connected && stream.CanRead)
                {
                    // Debug.Log("Receiving data...");
                    // Debug.Log($"{m_NetworkRunning}, {client.Connected}, {stream.CanRead}");
                    int length = IPAddress.NetworkToHostOrder(reader.ReadInt32()); // get file size
                    Debug.Log($"received filesize: {length}");
                    byte[] data = reader.ReadBytes(length); // get data
                    Debug.Log($"read {length} bytes");
                    // byte[] data = new byte[client.ReceiveBufferSize];
                    // int bytesRead = stream.Read(data, 0, client.ReceiveBufferSize);

                    // Debug.Log($"{data}");
                    dataQueue.Enqueue(data);
                }
            }
            catch { }
        }
    }

    public Material mat;
    public Texture2D tex = null;

    void Update()
    {
        if (dataQueue.Count > 0 && dataQueue.TryDequeue(out byte[] data))
        {
            if (tex == null)
                tex = new Texture2D(1, 1);
            bool isLoaded = tex.LoadImage(data);
            if (isLoaded)
            {
                tex.Apply();
                mat.mainTexture = tex;
            }
            else
            {
                Debug.LogError("Failed to load image data into texture.");
            }
        }
    }
}