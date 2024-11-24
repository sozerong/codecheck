import logo from './logo.svg';
import './App.css';
import backgroundImage from '../src/assets/images/railsbackground.png';
import check from '../src/assets/images/check.png';
import banner from '../src/assets/images/banner.png';
import sample from '../src/assets/images/sample.png';
import upload from '../src/assets/images/upload.png';
import hover from '../src/assets/images/hoverupload.png';
import before from '../src/assets/images/BeforeScan.png';

import styled from 'styled-components';
import MainBox from './Components/MainBox';
import PhotoUploader from './Components/PhotoUploader';

const BackImg = styled.div`
    background-image:url(${backgroundImage});
    background-repeat:no-repeat;
    background-size:cover;
    background-size:1920px 1080px;
    padding:0px 360px;

    height:100vw;
`
const StyledDiv = styled.div`
    display:flex;`
const CheckBox = styled.div`
  display:flex;
  margin:16px 0px 35.54px 0px;
`
const StyledP = styled.p`
  font-size:18px;
  color:#767676;
  margin:0;
  font-family:var(--font-regular);
  display:flex;
  align-items: center;
`
const CheckImg = styled.img`
  width:28.35px; height:28.35px;
  margin-right:16px;
`


function App() {
  return (
    <BackImg>
      <StyledDiv>
        <img src={sample} style={{width:'420.59px', height:'94.58px', margin:'63px auto 70.42px auto'}}></img>
      </StyledDiv>
      <StyledDiv>
          <MainBox text='이미지 업로드' firstImg={upload} secImg={hover}></MainBox>
          <MainBox text='이상감지' firstImg={before}></MainBox>
      </StyledDiv>
      <CheckBox>
        <CheckImg src={check}></CheckImg>
        <StyledP>이미지를 업로드하시면 서비스 약관에 동의하시는 것으로 간주하며, 철도로잉의 개인정보 처리방침 및 서비스 이용약관이 적용됩니다.</StyledP>
      </CheckBox>
      <img src={banner} style={{width:'1200px', height:'98px'}}></img>

      {/* <PhotoUploader></PhotoUploader> */}
      </BackImg>
      
  );
}

export default App;
