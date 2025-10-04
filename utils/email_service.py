"""
Email Service Integration
Handles sending emails through various providers (SMTP, SendGrid, etc.)
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import ssl
import httpx

from config.settings import settings


class EmailService:
    """
    Email service for sending marketing emails and notifications
    """

    def __init__(self):
        self.logger = logging.getLogger("email_service")
        self.provider = self._get_provider()

    def _get_provider(self) -> str:
        """Determine email provider from URL"""
        if not settings.EMAIL_SERVICE_URL:
            return "smtp"

        url = settings.EMAIL_SERVICE_URL.lower()
        if "sendgrid" in url:
            return "sendgrid"
        elif "mailgun" in url:
            return "mailgun"
        elif "ses" in url:
            return "ses"
        else:
            return "smtp"

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        html_body: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Send email to recipient

        Args:
            to_email: Recipient email
            subject: Email subject
            body: Plain text body
            from_email: Sender email (optional)
            html_body: HTML body (optional)
            **kwargs: Additional provider-specific options

        Returns:
            bool: True if successful
        """
        try:
            if self.provider == "sendgrid":
                return await self._send_sendgrid(
                    to_email, subject, body, from_email, html_body, **kwargs
                )
            elif self.provider == "mailgun":
                return await self._send_mailgun(
                    to_email, subject, body, from_email, html_body, **kwargs
                )
            elif self.provider == "ses":
                return await self._send_ses(
                    to_email, subject, body, from_email, html_body, **kwargs
                )
            else:
                return await self._send_smtp(
                    to_email, subject, body, from_email, html_body, **kwargs
                )

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}", exc_info=True)
            return False

    async def send_bulk_email(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        html_body: Optional[str] = None,
        **kwargs
    ) -> Dict[str, bool]:
        """
        Send email to multiple recipients

        Args:
            recipients: List of recipient emails
            subject: Email subject
            body: Plain text body
            from_email: Sender email (optional)
            html_body: HTML body (optional)

        Returns:
            Dict mapping email to success status
        """
        results = {}
        for email in recipients:
            success = await self.send_email(
                email, subject, body, from_email, html_body, **kwargs
            )
            results[email] = success

        return results

    async def _send_smtp(
        self,
        to_email: str,
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        html_body: Optional[str] = None,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Send email via SMTP"""
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = from_email or "noreply@marketing-system.com"
            msg["To"] = to_email

            # Add plain text body
            text_part = MIMEText(body, "plain")
            msg.attach(text_part)

            # Add HTML body if provided
            if html_body:
                html_part = MIMEText(html_body, "html")
                msg.attach(html_part)

            # Connect to SMTP server
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # Enable TLS

            # Login if credentials provided
            if username and password:
                server.login(username, password)

            # Send email
            server.sendmail(msg["From"], to_email, msg.as_string())
            server.quit()

            self.logger.info(f"Email sent via SMTP to {to_email}")
            return True

        except Exception as e:
            self.logger.error(f"SMTP send failed: {e}")
            return False

    async def _send_sendgrid(
        self,
        to_email: str,
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        html_body: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Send email via SendGrid"""
        try:
            url = "https://api.sendgrid.com/v3/mail/send"
            headers = {
                "Authorization": f"Bearer {api_key or self._get_api_key()}",
                "Content-Type": "application/json"
            }

            payload = {
                "personalizations": [{
                    "to": [{"email": to_email}],
                    "subject": subject
                }],
                "from": {"email": from_email or "noreply@marketing-system.com"},
                "content": []
            }

            # Add plain text
            payload["content"].append({
                "type": "text/plain",
                "value": body
            })

            # Add HTML if provided
            if html_body:
                payload["content"].append({
                    "type": "text/html",
                    "value": html_body
                })

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

            self.logger.info(f"Email sent via SendGrid to {to_email}")
            return True

        except Exception as e:
            self.logger.error(f"SendGrid send failed: {e}")
            return False

    async def _send_mailgun(
        self,
        to_email: str,
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        html_body: Optional[str] = None,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Send email via Mailgun"""
        try:
            if not domain:
                # Extract domain from URL or use default
                domain = "yourdomain.com"

            url = f"https://api.mailgun.net/v3/{domain}/messages"
            auth = ("api", api_key or self._get_api_key())

            data = {
                "from": from_email or f"noreply@{domain}",
                "to": to_email,
                "subject": subject,
                "text": body
            }

            if html_body:
                data["html"] = html_body

            async with httpx.AsyncClient() as client:
                response = await client.post(url, auth=auth, data=data)
                response.raise_for_status()

            self.logger.info(f"Email sent via Mailgun to {to_email}")
            return True

        except Exception as e:
            self.logger.error(f"Mailgun send failed: {e}")
            return False

    async def _send_ses(
        self,
        to_email: str,
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        html_body: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "us-east-1",
        **kwargs
    ) -> bool:
        """Send email via AWS SES"""
        try:
            import boto3
            from botocore.exceptions import ClientError

            # Create SES client
            ses_client = boto3.client(
                'ses',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )

            # Prepare message
            message = {
                'Subject': {'Data': subject},
                'Body': {'Text': {'Data': body}}
            }

            if html_body:
                message['Body']['Html'] = {'Data': html_body}

            # Send email
            response = ses_client.send_email(
                Source=from_email or "noreply@marketing-system.com",
                Destination={'ToAddresses': [to_email]},
                Message=message
            )

            self.logger.info(f"Email sent via SES to {to_email}")
            return True

        except Exception as e:
            self.logger.error(f"SES send failed: {e}")
            return False

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment"""
        # This should be set in environment variables
        return None  # Implement based on your setup

    async def get_delivery_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get delivery status for a sent email (provider-specific)

        Args:
            message_id: Message ID from send response

        Returns:
            Status information or None
        """
        # Implementation depends on provider
        # For now, return basic info
        return {
            "message_id": message_id,
            "status": "sent",
            "delivered": True
        }


class SMSService:
    """
    SMS service for sending text messages through various providers
    """

    def __init__(self):
        self.logger = logging.getLogger("sms_service")
        self.provider = self._get_provider()

    def _get_provider(self) -> str:
        """Determine SMS provider from URL"""
        if not settings.SMS_SERVICE_URL:
            return "twilio"

        url = settings.SMS_SERVICE_URL.lower()
        if "twilio" in url:
            return "twilio"
        elif "aws" in url or "sns" in url:
            return "sns"
        else:
            return "twilio"

    async def send_sms(
        self,
        to_phone: str,
        message: str,
        from_phone: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Send SMS to recipient

        Args:
            to_phone: Recipient phone number
            message: SMS message
            from_phone: Sender phone number (optional)

        Returns:
            bool: True if successful
        """
        try:
            if self.provider == "twilio":
                return await self._send_twilio(to_phone, message, from_phone, **kwargs)
            elif self.provider == "sns":
                return await self._send_sns(to_phone, message, from_phone, **kwargs)
            else:
                return await self._send_twilio(to_phone, message, from_phone, **kwargs)

        except Exception as e:
            self.logger.error(f"Failed to send SMS: {e}", exc_info=True)
            return False

    async def send_bulk_sms(
        self,
        recipients: List[str],
        message: str,
        from_phone: Optional[str] = None,
        **kwargs
    ) -> Dict[str, bool]:
        """
        Send SMS to multiple recipients

        Args:
            recipients: List of phone numbers
            message: SMS message
            from_phone: Sender phone number (optional)

        Returns:
            Dict mapping phone to success status
        """
        results = {}
        for phone in recipients:
            success = await self.send_sms(phone, message, from_phone, **kwargs)
            results[phone] = success

        return results

    async def _send_twilio(
        self,
        to_phone: str,
        message: str,
        from_phone: Optional[str] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Send SMS via Twilio"""
        try:
            from twilio.rest import Client

            client = Client(account_sid, auth_token)

            message = client.messages.create(
                body=message,
                from_=from_phone or "+1234567890",  # Replace with your Twilio number
                to=to_phone
            )

            self.logger.info(f"SMS sent via Twilio to {to_phone}")
            return True

        except Exception as e:
            self.logger.error(f"Twilio SMS send failed: {e}")
            return False

    async def _send_sns(
        self,
        to_phone: str,
        message: str,
        from_phone: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "us-east-1",
        **kwargs
    ) -> bool:
        """Send SMS via AWS SNS"""
        try:
            import boto3

            sns_client = boto3.client(
                'sns',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )

            response = sns_client.publish(
                PhoneNumber=to_phone,
                Message=message
            )

            self.logger.info(f"SMS sent via SNS to {to_phone}")
            return True

        except Exception as e:
            self.logger.error(f"SNS SMS send failed: {e}")
            return False


# Global service instances
email_service = EmailService()
sms_service = SMSService()
